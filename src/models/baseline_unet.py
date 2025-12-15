"""
Baseline U-Net model for comparison.
Standard U-Net architecture without transformer encoder or attention mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_blocks import DoubleConv, UpConv


class BaselineUNet(nn.Module):
    """
    Standard U-Net architecture for landmark detection.
    """
    def __init__(self, num_landmarks=19, in_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder (Contracting path)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        # Decoder (Expansive path)
        self.up4 = UpConv(base_channels * 16, base_channels * 8)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        
        self.up3 = UpConv(base_channels * 8, base_channels * 4)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.up2 = UpConv(base_channels * 4, base_channels * 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.up1 = UpConv(base_channels * 2, base_channels)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
        # Output layer
        self.final = nn.Conv2d(base_channels, num_landmarks, kernel_size=1)
        
        self.num_landmarks = num_landmarks
    
    def forward(self, x, return_aux=False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([F.interpolate(e4, size=d4.shape[2:], mode='bilinear', align_corners=False), d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False), d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False), d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False), d1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        if return_aux:
            # Return None for aux outputs to maintain compatibility
            return out, None, None, None
        
        return out

