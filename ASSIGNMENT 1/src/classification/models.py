"""
Model architectures and utilities for PCB Component Classification.
Supports:
  - EfficientNet-B0 / B2
  - MobileNetV3-Large
  - ConvNeXt-Tiny
  - ResNet34 / ResNet50
  - Parameter counting & Efficiency metric calculation
"""

from typing import Tuple
import math
import torch
import torch.nn as nn
from torchvision import models


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in the model.
    Returns: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def compute_efficiency_factor(num_trainable_params: int, median_params: int = None) -> float:
    """
    Compute model efficiency factor according to the assignment formula:
      E_i = clip(1 + 0.05 * [log10(P_med) - log10(P_i)], 0.90, 1.10)
    """
    if median_params is None:
        raise ValueError('Cross-submission median is unknown; supply an explicitly assumed median')
    if median_params <= 0:
        raise ValueError('Median must be positive')
    if num_trainable_params <= 0:
        return 1.0
    val = 1.0 + 0.05 * (math.log10(median_params) - math.log10(num_trainable_params))
    return float(max(0.90, min(1.10, val)))


def build_classifier(
    model_name: str = "efficientnet_b0",
    num_classes: int = 25,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    Build and return a classification model with the target number of output classes.
    """
    model_name = model_name.lower()

    if "efficientnet_b0" in model_name:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif "efficientnet_b2" in model_name:
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif "mobilenet_v3_large" in model_name:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif "convnext_tiny" in model_name:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif "resnet34" in model_name:
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif "resnet50" in model_name:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    return model

