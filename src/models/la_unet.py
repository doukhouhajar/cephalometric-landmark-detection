import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import DoubleConv, UpConv
from .attention_gate import AttentionGate
from .cbam import CBAM
from .swin_encoder import SwinEncoder


class LA_UNet(nn.Module):
    def __init__(self, num_landmarks=19):
        super().__init__()

        # Swin Transformer Encoder
        self.encoder = SwinEncoder(model_name="swin_tiny_patch4_window7_224")
        C1, C2, C3, C4 = self.encoder.out_channels  # ex: [96, 192, 384, 768]

        # Project Swin features -> UNet channel sizes
        self.proj1 = nn.Conv2d(C1, 64, kernel_size=1)
        self.proj2 = nn.Conv2d(C2, 128, kernel_size=1)
        self.proj3 = nn.Conv2d(C3, 256, kernel_size=1)
        self.proj4 = nn.Conv2d(C4, 512, kernel_size=1)

        # Bottleneck (CBAM)
        self.conv5 = DoubleConv(512, 1024)
        self.cbam = CBAM(1024)

        # Decoder + Attention Gates
        self.up4 = UpConv(1024, 512)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = UpConv(512, 256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = UpConv(256, 128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = UpConv(128, 64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64)

        # Output heatmaps (main output)
        self.final = nn.Conv2d(64, num_landmarks, kernel_size=1)

        # Multi-resolution deep supervision outputs
        self.ds4 = nn.Conv2d(512, num_landmarks, kernel_size=1)  # 1/4 resolution
        self.ds3 = nn.Conv2d(256, num_landmarks, kernel_size=1)  # 1/2 resolution
        self.ds2 = nn.Conv2d(128, num_landmarks, kernel_size=1)  # 1/2 resolution
        
        self.num_landmarks = num_landmarks

    def forward(self, x, return_aux=False):
        # Encoder (expects x to be 3-channel for Swin)
        f1, f2, f3, f4 = self.encoder(x)  # NCHW

        # Project channels
        c1 = self.proj1(f1)  # (B, 64, 56, 56)
        c2 = self.proj2(f2)  # (B, 128, 28, 28)
        c3 = self.proj3(f3)  # (B, 256, 14, 14)
        c4 = self.proj4(f4)  # (B, 512, 7, 7)

        # Bottleneck
        c5 = self.conv5(c4)     # (B, 1024, 7, 7)
        c5 = self.cbam(c5)

        # Decoder with alignment
        g4 = self.up4(c5)       # (B, 512, 14, 14)
        c4_att = self.att4(g4, c4)  # likely (B, 512, 7, 7) -> align
        c4_att = F.interpolate(c4_att, size=g4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([g4, c4_att], dim=1))

        g3 = self.up3(d4)       # (B, 256, 28, 28)
        c3_att = self.att3(g3, c3)  # (B, 256, 14, 14) -> align
        c3_att = F.interpolate(c3_att, size=g3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([g3, c3_att], dim=1))

        g2 = self.up2(d3)       # (B, 128, 56, 56)
        c2_att = self.att2(g2, c2)  # (B, 128, 28, 28) -> align
        c2_att = F.interpolate(c2_att, size=g2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([g2, c2_att], dim=1))

        g1 = self.up1(d2)       # (B, 64, 112, 112)
        c1_att = self.att1(g1, c1)  # (B, 64, 56, 56) -> align
        c1_att = F.interpolate(c1_att, size=g1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([g1, c1_att], dim=1))

        # Main output (full resolution)
        out = self.final(d1)    # (B, K, 112, 112) 
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        
        if return_aux:
            # Multi-resolution auxiliary outputs for deep supervision
            aux4 = self.ds4(d4)  # (B, K, 14, 14)
            aux4 = F.interpolate(aux4, size=x.shape[2:], mode="bilinear", align_corners=False)
            
            aux3 = self.ds3(d3)  # (B, K, 28, 28)
            aux3 = F.interpolate(aux3, size=x.shape[2:], mode="bilinear", align_corners=False)
            
            aux2 = self.ds2(d2)  # (B, K, 56, 56)
            aux2 = F.interpolate(aux2, size=x.shape[2:], mode="bilinear", align_corners=False)
            
            return out, aux2, aux3, aux4
        
        return out
