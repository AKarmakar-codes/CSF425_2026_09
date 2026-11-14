"""ResNet-18 Multi-Scale Backbone for Compact Mask2Former Student.

Conforms strictly to Assignment II Master Execution Plan (Architecture & Model Budget):
- ResNet-18 backbone (~11.2M parameters).
- Extracts 4-scale hierarchical feature maps: res2 (1/4), res3 (1/8), res4 (1/16), res5 (1/32).
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone with multi-scale feature tap (res2, res3, res4, res5)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        # Stage 0: stem
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Stage 1: res2 (stride 4, 64 channels)
        self.layer1 = base.layer1
        # Stage 2: res3 (stride 8, 128 channels)
        self.layer2 = base.layer2
        # Stage 3: res4 (stride 16, 256 channels)
        self.layer3 = base.layer3
        # Stage 4: res5 (stride 32, 512 channels)
        self.layer4 = base.layer4

        self.out_channels = {
            "res2": 64,
            "res3": 128,
            "res4": 256,
            "res5": 512,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass extracting multi-scale feature maps."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)   # (B, 64, H/4, W/4)
        c3 = self.layer2(c2)  # (B, 128, H/8, W/8)
        c4 = self.layer3(c3)  # (B, 256, H/16, W/16)
        c5 = self.layer4(c4)  # (B, 512, H/32, W/32)

        return {"res2": c2, "res3": c3, "res4": c4, "res5": c5}
