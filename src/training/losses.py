import torch
import torch.nn as nn
import torch.nn.functional as F
import piq   

# L = MSE(pred, target) + λ * (1 - SSIM)
class HeatmapLoss(nn.Module):
    def __init__(self, ssim_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - piq.ssim(pred, target, data_range=1.0)

        return mse_loss + self.ssim_weight * ssim_loss
