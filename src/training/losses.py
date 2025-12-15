# L = MSE(pred, target) + λ * (1 - SSIM) + Multi-resolution Deep Supervision
import torch
import torch.nn as nn
import torch.nn.functional as F
import piq
from src.training.metrics import soft_argmax_2d


def _compute_heatmap_loss(pred, target, mse_fn, ssim_weight=0.05):
    """Compute heatmap loss for a single prediction."""
    # Heatmap MSE
    mse_loss = mse_fn(pred, target)
    
    # SSIM (stabilisation)
    pred_sigmoid = torch.sigmoid(pred)
    target_clamped = torch.clamp(target, 0.0, 1.0)
    ssim_loss = 1 - piq.ssim(pred_sigmoid, target_clamped, data_range=1.0)
    
    return mse_loss + ssim_weight * ssim_loss


class HeatmapLoss(nn.Module):
    def __init__(self, ssim_weight=0.05, coord_weight=1.0, use_deep_supervision=True, 
                 deep_supervision_weights=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = ssim_weight
        self.coord_weight = coord_weight
        self.use_deep_supervision = use_deep_supervision
        
        # Weight for each deep supervision level (main, aux2, aux3, aux4)
        if deep_supervision_weights is None:
            deep_supervision_weights = [1.0, 0.5, 0.25, 0.125]
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
        total_loss = _compute_heatmap_loss(pred, target, self.mse, self.ssim_weight)
        
        # Deep supervision losses
        if self.use_deep_supervision and aux_outputs is not None:
            for i, aux_pred in enumerate(aux_outputs):
                if aux_pred is not None:
                    # Ensure aux prediction matches target resolution
                    if aux_pred.shape[2:] != target.shape[2:]:
                        aux_pred = F.interpolate(aux_pred, size=target.shape[2:], 
                                                mode='bilinear', align_corners=False)
                    aux_loss = _compute_heatmap_loss(aux_pred, target, self.mse, self.ssim_weight)
                    total_loss += self.ds_weights[i + 1] * aux_loss
        
        # Coordinate supervision (optional, for future use with proper coordinate transforms)
        if gt_coords is not None and self.coord_weight > 0:
            pred_coords = soft_argmax_2d(pred)
            coord_loss = torch.norm(pred_coords - gt_coords, dim=-1).mean()
            total_loss += self.coord_weight * coord_loss

        return total_loss
