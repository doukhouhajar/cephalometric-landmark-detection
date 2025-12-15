"""
Loss functions for heatmap-based landmark detection.
Implements Adaptive Wing Loss to handle class imbalance (99% background, 1% landmarks).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.training.metrics import soft_argmax_2d


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression.
    Handles class imbalance by focusing on foreground (landmark) regions.
    
    Reference: "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        
    def forward(self, pred, target):
        delta = (target - pred).abs()
        
        # Adaptive wing loss computation
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        
        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )
        
        return losses.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE that gives more importance to landmark regions.
    Weight = 1 + target * weight_factor (higher weight where heatmap > 0)
    """
    def __init__(self, weight_factor=50.0):
        super().__init__()
        self.weight_factor = weight_factor
        
    def forward(self, pred, target):
        # Create weight mask: higher weight for landmark regions
        weight = 1.0 + target * self.weight_factor
        
        # Weighted MSE
        loss = weight * (pred - target) ** 2
        return loss.mean()


class FocalMSELoss(nn.Module):
    """
    Focal-style MSE loss that focuses on hard examples (landmarks).
    """
    def __init__(self, gamma=2.0, weight_pos=10.0):
        super().__init__()
        self.gamma = gamma
        self.weight_pos = weight_pos
        
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        
        # Weight positive regions more
        weight = torch.where(target > 0.1, self.weight_pos, 1.0)
        
        # Focal weighting - focus on hard examples
        focal_weight = torch.pow(torch.abs(target - pred) + 1e-6, self.gamma)
        
        loss = weight * focal_weight * mse
        return loss.mean()


def _compute_heatmap_loss(pred, target, loss_fn):
    """Compute heatmap loss for a single prediction."""
    return loss_fn(pred, target)


class HeatmapLoss(nn.Module):
    def __init__(self, ssim_weight=0.05, coord_weight=1.0, use_deep_supervision=True, 
                 deep_supervision_weights=None, loss_type="awing"):
        super().__init__()
        
        # Choose loss function based on type
        if loss_type == "awing":
            self.loss_fn = AdaptiveWingLoss(omega=14, theta=0.5, epsilon=1, alpha=2.1)
        elif loss_type == "weighted_mse":
            self.loss_fn = WeightedMSELoss(weight_factor=50.0)
        elif loss_type == "focal_mse":
            self.loss_fn = FocalMSELoss(gamma=2.0, weight_pos=10.0)
        else:
            self.loss_fn = nn.MSELoss()
        
        self.coord_weight = coord_weight
        self.use_deep_supervision = use_deep_supervision
        
        # Weight for each deep supervision level (main, aux2, aux3, aux4)
        if deep_supervision_weights is None:
            deep_supervision_weights = [1.0, 0.4, 0.2, 0.1]
        self.ds_weights = deep_supervision_weights

    def forward(self, pred, target, gt_coords=None, aux_outputs=None):
        """
        Args:
            pred: Main prediction (B, K, H, W)
            target: Ground truth heatmaps (B, K, H, W)
            gt_coords: Optional ground truth coordinates (B, K, 2)
            aux_outputs: List of auxiliary predictions for deep supervision [(aux2, aux3, aux4)]
        """
        # Main output loss
        total_loss = _compute_heatmap_loss(pred, target, self.loss_fn)
        
        # Deep supervision losses
        if self.use_deep_supervision and aux_outputs is not None:
            for i, aux_pred in enumerate(aux_outputs):
                if aux_pred is not None:
                    # Ensure aux prediction matches target resolution
                    if aux_pred.shape[2:] != target.shape[2:]:
                        aux_pred = F.interpolate(aux_pred, size=target.shape[2:], 
                                                mode='bilinear', align_corners=False)
                    aux_loss = _compute_heatmap_loss(aux_pred, target, self.loss_fn)
                    total_loss += self.ds_weights[i + 1] * aux_loss
        
        # Coordinate supervision (optional)
        if gt_coords is not None and self.coord_weight > 0:
            pred_coords = soft_argmax_2d(pred)
            coord_loss = torch.norm(pred_coords - gt_coords, dim=-1).mean()
            total_loss += self.coord_weight * coord_loss

        return total_loss
