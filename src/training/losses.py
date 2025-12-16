import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.metrics import soft_argmax_2d


class AggressiveWeightedMSE(nn.Module):
    """
    Heavily weighted MSE - landmark pixels get 100x more weight.
    This forces the network to produce peaks, not uniform outputs.
    """
    def __init__(self, pos_weight=100.0, neg_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
    def forward(self, pred, target):
        # Binary mask for positive regions (where target > threshold)
        pos_mask = (target > 0.01).float()
        neg_mask = 1.0 - pos_mask
        
        # Compute squared error
        sq_error = (pred - target) ** 2
        
        # Apply different weights to positive vs negative regions
        weighted_loss = self.pos_weight * pos_mask * sq_error + self.neg_weight * neg_mask * sq_error
        
        return weighted_loss.mean()


class CombinedLandmarkLoss(nn.Module):
    """
    Combined loss that uses both:
    1. Weighted heatmap MSE (to produce peaks)
    2. Coordinate supervision (to place peaks correctly)
    """
    def __init__(self, heatmap_weight=1.0, coord_weight=5.0, pos_weight=100.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.heatmap_loss = AggressiveWeightedMSE(pos_weight=pos_weight)

    def forward(self, pred, target, gt_coords=None):
        # Heatmap loss
        heatmap_loss = self.heatmap_loss(pred, target)

        total_loss = self.heatmap_weight * heatmap_loss

        # Coordinate supervision - THIS IS KEY
        if gt_coords is not None:
            pred_coords = soft_argmax_2d(pred)
            # L1 loss on coordinates (more robust than L2)
            coord_loss = F.l1_loss(pred_coords, gt_coords)
            total_loss = total_loss + self.coord_weight * coord_loss
        
        return total_loss


class HeatmapLoss(nn.Module):
    def __init__(self, coord_weight=5.0, use_deep_supervision=True, 
                 deep_supervision_weights=None, loss_type="combined", **kwargs):
        super().__init__()
        
        # Use combined loss with coordinate supervision
        self.loss_fn = CombinedLandmarkLoss(
            heatmap_weight=1.0,
            coord_weight=coord_weight,
            pos_weight=100.0  # Heavily weight positive pixels
        )
        
        self.use_deep_supervision = use_deep_supervision
        self.coord_weight = coord_weight
        
        if deep_supervision_weights is None:
            deep_supervision_weights = [1.0, 0.4, 0.2, 0.1]
        self.ds_weights = deep_supervision_weights

    def forward(self, pred, target, gt_coords=None, aux_outputs=None):
        """
        Args:
            pred: Main prediction (B, K, H, W)
            target: Ground truth heatmaps (B, K, H, W)
            gt_coords: Ground truth coordinates (B, K, 2) - NOW REQUIRED
            aux_outputs: List of auxiliary predictions for deep supervision
        """
        # Main output loss WITH coordinate supervision
        total_loss = self.loss_fn(pred, target, gt_coords)
        
        # Deep supervision losses
        if self.use_deep_supervision and aux_outputs is not None:
            for i, aux_pred in enumerate(aux_outputs):
                if aux_pred is not None:
                    if aux_pred.shape[2:] != target.shape[2:]:
                        aux_pred = F.interpolate(aux_pred, size=target.shape[2:], 
                                                mode='bilinear', align_corners=False)
                    # Also use coordinate supervision for auxiliary outputs
                    aux_loss = self.loss_fn(aux_pred, target, gt_coords)
                    total_loss += self.ds_weights[i + 1] * aux_loss

        return total_loss
