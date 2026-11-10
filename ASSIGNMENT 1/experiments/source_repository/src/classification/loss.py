"""
Loss functions designed for imbalanced classification in PCB components.
Implements:
  - FocalLoss: focuses on hard-to-classify and minority samples
  - LabelSmoothingCrossEntropy: prevents overconfidence on rare classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import numpy as np


def compute_class_balanced_weights(
    samples: List[Dict],
    num_classes: int = 25,
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Compute Class-Balanced Loss weights based on effective number of samples:
        E_n = (1 - beta^n) / (1 - beta)
        W_c = (1 - beta) / (1 - beta^{n_c}) = 1 / E_{n_c}
    Reference: Cui et al., 'Class-Balanced Loss Based on Effective Number of Samples', CVPR 2019.
    Weights are re-scaled so sum(weights) = num_classes (mean weight is 1.0).
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in samples:
        cid = int(s["class_id"]) - 1
        if 0 <= cid < num_classes:
            counts[cid] += 1.0

    effective_num = 1.0 - np.power(beta, counts)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0 and effective_num[c] > 0:
            weights[c] = float((1.0 - beta) / effective_num[c])
        else:
            weights[c] = 0.0

    total_w = float(np.sum(weights))
    if total_w > 0:
        weights = (weights / total_w) * num_classes

    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al., CVPR 2019).
    Modulates Focal Loss with class weights derived from effective sample numbers:
        CB_FL(p_t) = -W_y * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        samples: Optional[List[Dict]] = None,
        weights: Optional[torch.Tensor] = None,
        num_classes: int = 25,
        beta: float = 0.999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if weights is not None:
            self.weights = weights
        elif samples is not None:
            self.weights = compute_class_balanced_weights(samples, num_classes=num_classes, beta=beta)
        else:
            self.weights = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        targets_unsqueezed = targets.unsqueeze(1)
        log_pt = log_p.gather(1, targets_unsqueezed).squeeze(1)
        pt = p.gather(1, targets_unsqueezed).squeeze(1)

        focal_weight = torch.pow(1.0 - pt, self.gamma)

        if self.weights is not None:
            if self.weights.device != inputs.device:
                self.weights = self.weights.to(inputs.device)
            w = self.weights.gather(0, targets)
            loss = -w * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, num_classes) - raw unnormalized logits
        targets: (batch_size,) - ground truth class indices [0..num_classes-1]
        """
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        # Gather target log-probabilities and probabilities
        targets_unsqueezed = targets.unsqueeze(1)
        log_pt = log_p.gather(1, targets_unsqueezed).squeeze(1)
        pt = p.gather(1, targets_unsqueezed).squeeze(1)

        focal_weight = torch.pow(1.0 - pt, self.gamma)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with label smoothing for regularizing training.
    """

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)

        smooth_loss = -logprobs.mean(dim=-1)
        nll_loss = F.nll_loss(logprobs, targets, weight=self.weight, reduction="none")
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

