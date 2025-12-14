import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import DoubleConv, UpConv
from .attention_gate import AttentionGate
from .cbam import CBAM
from .swin_encoder import SwinEncoder

class LA_UNet(nn.Module):
    def __init__(self, num_landmarks=19, input_channels=1, feature_size=64):
        super().__init__()
        # Encoder
        self.encoder = SwinEncoder(model_name="swin_tiny_patch4_window7_224")
        C1, C2, C3, C4 = self.encoder.out_channels  
        self.proj1 = nn.Conv2d(C1, 64, kernel_size=1)
        self.proj2 = nn.Conv2d(C2, 128, kernel_size=1)
        self.proj3 = nn.Conv2d(C3, 256, kernel_size=1)
        self.proj4 = nn.Conv2d(C4, 512, kernel_size=1)

        # Bottleneck
        self.conv5 = DoubleConv(feature_size * 8, feature_size * 16)
        self.cbam = CBAM(feature_size * 16)

        # Decoder + Attention Gates
        self.att4 = AttentionGate(F_g=feature_size * 16, F_l=feature_size * 8, F_int=feature_size * 4)
        self.up4 = UpConv(feature_size * 16, feature_size * 8)
        self.dec4 = DoubleConv(feature_size * 16, feature_size * 8)

        self.att3 = AttentionGate(F_g=feature_size * 8, F_l=feature_size * 4, F_int=feature_size * 2)
        self.up3 = UpConv(feature_size * 8, feature_size * 4)
        self.dec3 = DoubleConv(feature_size * 8, feature_size * 4)

        self.att2 = AttentionGate(F_g=feature_size * 4, F_l=feature_size * 2, F_int=feature_size)
        self.up2 = UpConv(feature_size * 4, feature_size * 2)
        self.dec2 = DoubleConv(feature_size * 4, feature_size * 2)

        self.att1 = AttentionGate(F_g=feature_size * 2, F_l=feature_size, F_int=feature_size // 2)
        self.up1 = UpConv(feature_size * 2, feature_size)
        self.dec1 = DoubleConv(feature_size * 2, feature_size)

        # Output heatmaps
        self.final = nn.Conv2d(feature_size, num_landmarks, kernel_size=1)

    def forward(self, x):
        # Encoder
        f1, f2, f3, f4 = self.encoder(x)   # resolutions: 1/4, 1/8, 1/16, 1/32

        # Project channels
        c1 = self.proj1(f1)   # 64 channels
        c2 = self.proj2(f2)   # 128 channels
        c3 = self.proj3(f3)   # 256 channels
        c4 = self.proj4(f4)   # 512 channels

        # Bottleneck
        c5 = self.conv5(c4)
        c5 = self.cbam(c5)

        # Decoder + Attention
        g4 = self.up4(c5)
        c4_att = self.att4(g4, c4)
        d4 = self.dec4(torch.cat([g4, c4_att], dim=1))

        g3 = self.up3(d4)
        c3_att = self.att3(g3, c3)
        d3 = self.dec3(torch.cat([g3, c3_att], dim=1))

        g2 = self.up2(d3)
        c2_att = self.att2(g2, c2)
        d2 = self.dec2(torch.cat([g2, c2_att], dim=1))

        g1 = self.up1(d2)
        c1_att = self.att1(g1, c1)
        d1 = self.dec1(torch.cat([g1, c1_att], dim=1))

        # Output heatmaps
        out = self.final(d1)

        return out
