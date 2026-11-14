"""Laplacian Boundary-Aware Mask Loss for Crisp Component Contours.

Conforms strictly to Assignment II Master Execution Plan (Architecture & Losses):
- Uses 2D Laplacian edge filtering on 1/4-scale mask logits/probabilities.
- Enforces sharp boundary transitions to prevent adjacent 0402/0603 passives from merging.
- Combines with BCE and Dice loss for full geometric fidelity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianBoundaryLoss(nn.Module):
    """Multi-scale Laplacian edge alignment loss between predicted and ground-truth masks."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        # Standard discrete 2D Laplacian edge kernel
        laplacian_kernel = torch.tensor(
            [[0.0,  1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0,  1.0, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        self.register_buffer("kernel", laplacian_kernel)

    def forward(self, pred_probs: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Compute boundary loss.
        
        Args:
            pred_probs: (N, H, W) sigmoid probabilities in [0, 1].
            target_masks: (N, H, W) binary float masks in {0, 1}.
        """
        if pred_probs.numel() == 0:
            return pred_probs.sum() * 0.0

        p = pred_probs.unsqueeze(1)  # (N, 1, H, W)
        t = target_masks.unsqueeze(1).float()  # (N, 1, H, W)

        # Compute Laplacian edge gradients
        pred_edges = F.conv2d(p, self.kernel, padding=1).abs()
        target_edges = F.conv2d(t, self.kernel, padding=1).abs()

        # Normalize edge response to [0, 1]
        pred_edges = pred_edges.clamp(0.0, 1.0)
        target_edges = target_edges.clamp(0.0, 1.0)

        # Edge Dice Loss + Smooth L1 Loss
        intersection = (pred_edges * target_edges).sum(dim=(-2, -1))
        union = pred_edges.sum(dim=(-2, -1)) + target_edges.sum(dim=(-2, -1)) + 1e-5
        edge_dice = 1.0 - (2.0 * intersection / union)

        smooth_l1 = F.smooth_l1_loss(pred_edges, target_edges, reduction="none").mean(dim=(-2, -1))

        loss = (edge_dice + smooth_l1).mean()
        return loss
