"""
Loss functions designed for imbalanced classification in PCB components.
Implements:
  - FocalLoss: focuses on hard-to-classify and minority samples
  - LabelSmoothingCrossEntropy: prevents overconfidence on rare classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
        self.register_buffer('alpha', alpha)
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

