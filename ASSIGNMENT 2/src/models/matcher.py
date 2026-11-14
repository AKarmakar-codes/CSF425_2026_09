"""Hungarian Bipartite Matcher for Mask2Former Training.

Conforms strictly to Assignment II Master Execution Plan (Step 3 & 4):
- Matches predicted queries to ground-truth instances using the Hungarian algorithm.
- Cost = class cost + mask focal/BCE cost + mask dice cost.
- Operates on the final decoder layer predictions (or any auxiliary layer).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """Hungarian bipartite matcher for query-to-GT instance assignment.

    For each image in the batch, finds the minimum-cost one-to-one matching
    between N predicted queries and M ground-truth instances.

    Cost matrix: C = w_cls * C_cls + w_mask * C_mask + w_dice * C_dice
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute bipartite matching between predictions and targets.

        Args:
            pred_logits: (B, Q, C+1) class logits.
            pred_masks:  (B, Q, H, W) mask logits (un-sigmoided, at 1/4 scale).
            targets: list of dicts per batch image, each with:
                - "labels": (M,) int64 ground-truth class IDs (1..25, 0=background)
                - "masks":  (M, H, W) float binary masks at same scale as pred_masks

        Returns:
            List of (query_indices, gt_indices) tuples, one per batch image.
        """
        B, Q, _ = pred_logits.shape
        indices = []

        for b in range(B):
            tgt_labels = targets[b]["labels"]   # (M,)
            tgt_masks  = targets[b]["masks"]    # (M, H, W)
            M = len(tgt_labels)

            if M == 0:
                # No GT instances: no matches
                indices.append((
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                ))
                continue

            # 1. Classification cost: negative log-probability of GT class
            out_prob = pred_logits[b].softmax(-1)   # (Q, C+1)
            # tgt_labels are 1..25; index directly into class probability
            cost_class = -out_prob[:, tgt_labels]   # (Q, M)

            # 2. Mask focal/BCE cost (subsample pixels for efficiency)
            out_mask = pred_masks[b]                # (Q, H, W)
            H, W = out_mask.shape[-2:]
            # Flatten and compute BCE on sigmoid
            out_mask_sig = out_mask.sigmoid().flatten(1)    # (Q, HW)
            tgt_flat = tgt_masks.flatten(1).float()         # (M, HW)
            # Binary cross entropy per pair: outer product approximation
            cost_mask = self._batch_sigmoid_focal_loss(out_mask_sig, tgt_flat)  # (Q, M)

            # 3. Dice cost
            cost_dice = self._batch_dice_loss(out_mask_sig, tgt_flat)  # (Q, M)

            # Combined cost matrix
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )  # (Q, M)

            # Solve assignment on CPU (scipy)
            C_np = C.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_np)

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))

        return indices

    @staticmethod
    def _batch_sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Pairwise sigmoid focal loss. inputs: (Q, N), targets: (M, N) → (Q, M)."""
        # Compute for each (q, m) pair via broadcasting
        inputs = inputs.unsqueeze(1)    # (Q, 1, N)
        targets = targets.unsqueeze(0)  # (1, M, N)
        p = inputs.sigmoid()
        ce = F.binary_cross_entropy_with_logits(inputs, targets.expand_as(inputs), reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** gamma * ce
        return focal.mean(-1)   # (Q, M)

    @staticmethod
    def _batch_dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise dice loss. inputs: (Q, N) sigmoid, targets: (M, N) binary → (Q, M)."""
        inputs = inputs.unsqueeze(1)    # (Q, 1, N)
        targets = targets.unsqueeze(0)  # (1, M, N)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1) + 1e-5
        return 1.0 - numerator / denominator  # (Q, M)
