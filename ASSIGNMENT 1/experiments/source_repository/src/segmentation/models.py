"""
Instance segmentation architectures and parameter efficiency utilities.
Supports:
  - Mask R-CNN with ResNet-50-FPN backbone (25 classes + 1 background)
  - Custom lightweight mask heads
  - Trainable parameter counting and efficiency factor calculation
"""

from typing import Tuple
import math
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def compute_efficiency_factor(num_trainable_params: int, median_params: int = 44_000_000) -> float:
    """
    Compute segmentation efficiency factor according to the assignment formula:
      E_i = clip(1 + 0.05 * [log10(P_med) - log10(P_i)], 0.90, 1.10)
    """
    if num_trainable_params <= 0:
        return 1.0
    val = 1.0 + 0.05 * (math.log10(median_params) - math.log10(num_trainable_params))
    return float(max(0.90, min(1.10, val)))


def build_maskrcnn(
    num_classes: int = 26,  # 25 classes + 1 background
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    """
    Build Mask R-CNN model adapted for the 25 PCB component classes (+ background = 26).
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # 1. Replace the box classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    return model

