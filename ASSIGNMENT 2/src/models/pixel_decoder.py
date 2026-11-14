"""Multi-Scale Feature Pyramid Pixel Decoder for Mask2Former.

Conforms strictly to Assignment II Master Execution Plan (Architecture):
- Top-down feature fusion fusing res5 (1/32), res4 (1/16), res3 (1/8) to hidden dim D=256.
- High-resolution 1/4-scale mask feature fusion with res2.
- Provides hooked feature maps for true differentiable Grad-CAM.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Standard Conv2d + GroupNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))


class FPNPixelDecoder(nn.Module):
    """FPN-based Pixel Decoder producing multi-scale features and 1/4-scale mask features."""

    def __init__(
        self,
        in_channels: Dict[str, int],
        hidden_dim: int = 256,
        mask_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim

        # Lateral 1x1 convolutions for top-down pathway
        self.lateral_c5 = nn.Conv2d(in_channels["res5"], hidden_dim, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels["res4"], hidden_dim, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(in_channels["res3"], hidden_dim, kernel_size=1)
        self.lateral_c2 = nn.Conv2d(in_channels["res2"], hidden_dim, kernel_size=1)

        # Output smooth 3x3 convolutions
        self.output_p5 = ConvModule(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.output_p4 = ConvModule(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.output_p3 = ConvModule(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # High-resolution 1/4 mask feature projection
        self.mask_features_conv = ConvModule(hidden_dim, mask_dim, kernel_size=3, padding=1)

        # Cache for Grad-CAM hook
        self.last_mask_features = None

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        Returns:
            mask_features: (B, mask_dim, H/4, W/4) for dot-product mask generation and Grad-CAM.
            multi_scale_features: [p3 (1/8), p4 (1/16), p5 (1/32)] for Transformer masked attention.
        """
        c2 = features["res2"]  # (B, 64, H/4, W/4)
        c3 = features["res3"]  # (B, 128, H/8, W/8)
        c4 = features["res4"]  # (B, 256, H/16, W/16)
        c5 = features["res5"]  # (B, 512, H/32, W/32)

        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p5_out = self.output_p5(p5)
        p4_out = self.output_p4(p4)
        p3_out = self.output_p3(p3)

        # 1/4 scale mask features
        mask_features = self.mask_features_conv(p2)
        self.last_mask_features = mask_features  # for Grad-CAM hook

        return mask_features, [p3_out, p4_out, p5_out]
