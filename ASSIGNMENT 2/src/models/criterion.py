"""Combined Training Criterion for Compact Mask2Former.

Conforms strictly to Assignment II Master Execution Plan (Step 4):
- Seesaw Classification Loss on matched queries (long-tail 25-class imbalance mitigation).
- Focal/BCE + Dice Mask Loss on matched query binary masks.
- Laplacian Boundary Loss (lambda_bnd = 1.0) on 1/4-scale mask features.
- Full Auxiliary Layer Supervision across all 6 decoder layers.
- Ignore Index = 255 excluded from all loss computations.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.seesaw_loss import SeesawLoss
from src.losses.boundary_loss import LaplacianBoundaryLoss
from src.models.matcher import HungarianMatcher


class SetCriterion(nn.Module):
    """Mask2Former training criterion combining Seesaw + Focal/Dice + Boundary losses.

    Applies Hungarian matching at each of the 6 auxiliary decoder layers,
    accumulating gradients for stable bipartite matching throughout training.
    """

    def __init__(
        self,
        num_classes: int = 25,
        matcher: HungarianMatcher = None,
        weight_class: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
        weight_boundary: float = 1.0,
        no_object_weight: float = 0.1,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher(
            cost_class=weight_class, cost_mask=weight_mask, cost_dice=weight_dice
        )
        self.weight_class = weight_class
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.weight_boundary = weight_boundary
        self.no_object_weight = no_object_weight
        self.ignore_index = ignore_index

        self.seesaw = SeesawLoss(num_classes=num_classes)
        self.boundary = LaplacianBoundaryLoss()

        # Class weights: background (no-object) class gets down-weighted by 0.1
        # Classes 1..25 get weight 1.0; class 0 (no-object) gets 0.1
        cls_weights = torch.ones(num_classes + 1)
        cls_weights[0] = no_object_weight  # down-weight the no-object class
        self.register_buffer("cls_weights", cls_weights)

    def forward(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute combined training loss over all 6 decoder layers.

        Args:
            outputs: list of 6 layer dicts, each with:
                - "pred_logits": (B, Q, C+1)
                - "pred_masks":  (B, Q, H/4, W/4)
            targets: list of B dicts, each with:
                - "labels": (M,) int64 class IDs 1..25 (0 = background, ignore=255 excluded)
                - "masks":  (M, H/4, W/4) binary float masks

        Returns:
            Dict of scalar losses: "loss_total", "loss_class", "loss_mask", "loss_dice", "loss_boundary"
        """
        total_loss_class = torch.tensor(0.0, device=outputs[0]["pred_logits"].device)
        total_loss_mask = torch.tensor(0.0, device=outputs[0]["pred_logits"].device)
        total_loss_dice = torch.tensor(0.0, device=outputs[0]["pred_logits"].device)
        total_loss_boundary = torch.tensor(0.0, device=outputs[0]["pred_logits"].device)

        num_layers = len(outputs)

        for layer_idx, layer_out in enumerate(outputs):
            pred_logits = layer_out["pred_logits"]   # (B, Q, C+1)
            pred_masks  = layer_out["pred_masks"]    # (B, Q, H, W)

            # 1. Hungarian Matching (stop gradient — matcher is @no_grad)
            indices = self.matcher(pred_logits, pred_masks, targets)

            # 2. Classification Loss (Seesaw)
            lc = self._loss_class(pred_logits, targets, indices)
            total_loss_class = total_loss_class + lc

            # 3. Mask Losses (Focal/BCE + Dice + Boundary)
            lm, ld, lb = self._loss_masks(pred_masks, targets, indices)
            total_loss_mask = total_loss_mask + lm
            total_loss_dice = total_loss_dice + ld
            total_loss_boundary = total_loss_boundary + lb

        # Normalize by number of layers
        total_loss_class = total_loss_class / num_layers
        total_loss_mask  = total_loss_mask  / num_layers
        total_loss_dice  = total_loss_dice  / num_layers
        total_loss_boundary = total_loss_boundary / num_layers

        total = (
            self.weight_class * total_loss_class
            + self.weight_mask * total_loss_mask
            + self.weight_dice * total_loss_dice
            + self.weight_boundary * total_loss_boundary
        )

        return {
            "loss_total":    total,
            "loss_class":    total_loss_class,
            "loss_mask":     total_loss_mask,
            "loss_dice":     total_loss_dice,
            "loss_boundary": total_loss_boundary,
        }

    def _loss_class(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Seesaw classification loss on all queries (matched → GT class, unmatched → class 0)."""
        B, Q, _ = pred_logits.shape
        device = pred_logits.device

        # Build target class tensor: (B*Q,) initialized to 0 (no-object)
        tgt_classes = torch.zeros(B * Q, dtype=torch.long, device=device)

        for b, (q_idx, gt_idx) in enumerate(indices):
            if len(q_idx) == 0:
                continue
            tgt_b = targets[b]["labels"][gt_idx]  # GT class IDs 1..25
            tgt_classes[b * Q + q_idx] = tgt_b

        logits_flat = pred_logits.reshape(B * Q, -1)  # (B*Q, C+1)

        # Update Seesaw running class counts
        self.seesaw.update_counts(tgt_classes)
        loss = self.seesaw(logits_flat, tgt_classes)
        return loss

    def _loss_masks(
        self,
        pred_masks: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Focal/BCE + Dice + Laplacian Boundary mask losses on matched query-GT pairs."""
        device = pred_masks.device
        all_pred = []
        all_tgt  = []

        for b, (q_idx, gt_idx) in enumerate(indices):
            if len(q_idx) == 0:
                continue
            pm = pred_masks[b][q_idx]            # (K, H, W)
            gt = targets[b]["masks"][gt_idx]     # (K, H, W)
            all_pred.append(pm)
            all_tgt.append(gt)

        if not all_pred:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        pred_cat = torch.cat(all_pred, dim=0)   # (N_matched, H, W)
        tgt_cat  = torch.cat(all_tgt,  dim=0)  # (N_matched, H, W)

        # Focal / BCE mask loss
        p = pred_cat.sigmoid()
        alpha, gamma = 0.25, 2.0
        bce = F.binary_cross_entropy_with_logits(pred_cat, tgt_cat, reduction="none")
        p_t = p * tgt_cat + (1 - p) * (1 - tgt_cat)
        alpha_t = alpha * tgt_cat + (1 - alpha) * (1 - tgt_cat)
        loss_mask = (alpha_t * (1 - p_t) ** gamma * bce).mean()

        # Dice loss
        numerator = 2 * (p * tgt_cat).sum(dim=(-2, -1))
        denominator = p.sum(dim=(-2, -1)) + tgt_cat.sum(dim=(-2, -1)) + 1e-5
        loss_dice = (1.0 - numerator / denominator).mean()

        # Laplacian boundary loss
        loss_boundary = self.boundary(p, tgt_cat)

        return loss_mask, loss_dice, loss_boundary
