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
        # Cast to float32 to prevent half-precision (fp16) arithmetic overflow in einsum/dot-products
        pred_logits = pred_logits.float()
        pred_masks  = pred_masks.float()
        indices = []

        for b in range(B):
            tgt_labels = targets[b]["labels"]   # (M,)
            tgt_masks  = targets[b]["masks"].float()  # (M, H, W)
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

            # 2. Mask focal/BCE cost and Dice cost via einsum (avoids giant QxMxHW VRAM allocation)
            out_mask = pred_masks[b]                # (Q, H, W)
            out_mask_flat = out_mask.flatten(1)     # (Q, HW) raw mask logits
            tgt_flat = tgt_masks.flatten(1)         # (M, HW) binary ground-truth

            cost_mask = self._batch_sigmoid_focal_loss(out_mask_flat, tgt_flat)  # (Q, M)
            cost_dice = self._batch_dice_loss(out_mask_flat, tgt_flat)           # (Q, M)

            # Combined cost matrix
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )  # (Q, M)

            # Ensure strictly finite cost matrix for scipy linear_sum_assignment (failsafe against NaN/Inf)
            C = torch.nan_to_num(C, nan=1e4, posinf=1e4, neginf=-1e4)

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
        """Pairwise sigmoid focal loss via einsum.
        inputs: (Q, HW) un-sigmoided mask logits
        targets: (M, HW) binary masks {0, 1}
        Returns: (Q, M) pairwise focal cost
        """
        inputs = inputs.float()
        targets = targets.float()
        hw = inputs.shape[1]
        prob = inputs.sigmoid()
        focal_pos = ((1.0 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )
        if alpha >= 0:
            focal_pos = focal_pos * alpha
            focal_neg = focal_neg * (1.0 - alpha)

        cost = (torch.einsum("qc,mc->qm", focal_pos, targets) + torch.einsum("qc,mc->qm", focal_neg, (1.0 - targets))) / hw
        return cost

    @staticmethod
    def _batch_dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise Dice loss via einsum.
        inputs: (Q, HW) un-sigmoided mask logits
        targets: (M, HW) binary masks {0, 1}
        Returns: (Q, M) pairwise Dice cost
        """
        inputs = inputs.float()
        targets = targets.float()
        prob = inputs.sigmoid()
        numerator = 2.0 * torch.einsum("qc,mc->qm", prob, targets)
        denominator = prob.sum(dim=-1)[:, None] + targets.sum(dim=-1)[None, :] + 1e-5
        return 1.0 - numerator / denominator
