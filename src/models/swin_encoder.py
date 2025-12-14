import torch
import torch.nn as nn
import timm


class SwinEncoder(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True  # important
        )
        # Output channels of each stage (Swin-Tiny)
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x):
        features = self.backbone(x)

        # timm Swin outputs features in NHWC -> convert to NCHW
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        return features
    """
  timm Swin Transformer output 4 feature maps:
    - stage1: 1/4 resolution
    - stage2: 1/8
    - stage3: 1/16
    - stage4: 1/32
    """