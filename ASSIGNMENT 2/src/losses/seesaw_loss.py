"""Seesaw Loss for 25-Class Long-Tail PCB Imbalance Mitigation.

Conforms strictly to Assignment II Master Execution Plan (Architecture & Losses):
- Replaces standard uniform cross-entropy with dynamic Seesaw Loss.
- Mitigation factor (p = 0.8): Reduces false negative gradients on rare classes (V, QA, M, CRA, etc.).
- Compensation factor (q = 2.0): Restores penalty when rare classes make genuine false positive errors.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeesawLoss(nn.Module):
    """Seesaw Loss for long-tail multi-class classification."""

    def __init__(
        self,
        num_classes: int = 25,
        p: float = 0.8,
        q: float = 2.0,
        eps: float = 1e-4,
        class_counts: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps

        # Cumulative frequency register (1-indexed for classes, 0 for background/no-object)
        if class_counts is not None:
            self.register_buffer("class_counts", class_counts.float())
        else:
            self.register_buffer("class_counts", torch.ones(num_classes + 1, dtype=torch.float))

    def update_counts(self, targets: torch.Tensor):
        """Update running class occurrence counts from batch targets."""
        with torch.no_grad():
            valid_targets = targets[targets >= 0]
            counts = torch.bincount(valid_targets, minlength=self.num_classes + 1).float()
            self.class_counts += counts

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Seesaw loss.
        
        Args:
            logits: (N, C + 1) prediction logits where class 0 is no-object / background.
            targets: (N,) target class IDs in [0, C] or -1 / 255 for ignore.
        """
        valid_mask = (targets >= 0) & (targets <= self.num_classes)
        if not valid_mask.any():
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        num_samples, num_classes_with_bg = logits.shape

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.unsqueeze(1))  # (N, 1)

        # 1. Mitigation factor: M_{ij} = (N_j / N_i)^p if N_j < N_i else 1.0
        # self.class_counts: (C + 1,)
        n_i = self.class_counts[targets].unsqueeze(1)  # (N, 1)
        n_j = self.class_counts.unsqueeze(0)  # (1, C + 1)
        ratio = (n_j / (n_i + self.eps)).clamp(max=1.0)
        mitigation = ratio.pow(self.p)  # (N, C + 1)

        # 2. Compensation factor: C_{ij} = (prob_j / prob_i)^q if prob_j > prob_i else 1.0
        prob_ratio = (probs / (target_probs + self.eps)).clamp(min=1.0)
        compensation = prob_ratio.pow(self.q)  # (N, C + 1)

        # Combined seesaw weight
        seesaw_weights = mitigation * compensation  # (N, C + 1)

        # Don't scale the ground truth target logit
        one_hot = F.one_hot(targets, num_classes=num_classes_with_bg).bool()
        seesaw_weights[one_hot] = 1.0

        # Numerically stable self-normalized logits
        scaled_logits = logits + (seesaw_weights.clamp(min=1e-3, max=100.0).log())
        loss = F.cross_entropy(scaled_logits, targets, reduction="mean")
        return loss
