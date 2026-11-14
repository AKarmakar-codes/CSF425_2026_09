"""Diagnostic evaluation against frozen Teacher Pseudo-Labels (Q1e & Q2d).

Conforms strictly to Assignment II Master Execution Plan (Step 2) & PDF Section 6:
- Computes pixel-level semantic metrics (F1^ps, mIoU^ps) on non-ignored pixels (ignore_index = 255).
- Background (0) correctly contributes to false positives when model hallucinates components.
- Computes instance-level metrics (AP_50^ps, mAP_50:95^ps) with class-aware bipartite IoU matching.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.constants import CLASSES, NUM_CLASSES, BACKGROUND_ID, IGNORE_ID, decode_source_mask


def compute_semantic_metrics(
    pred_sem: np.ndarray,
    target_sem: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_id: int = IGNORE_ID,
) -> Dict[str, Any]:
    """Calculate class-wise and macro P, R, F1, and mIoU on valid (non-ignored) pixels.
    
    Background (0) is treated as valid background:
    - Target 0 vs Pred >0 counts as False Positive (hallucination).
    - Pixels marked with ignore_id (255) in target are excluded from evaluation.
    """
    valid_mask = (target_sem != ignore_id)
    y_true = target_sem[valid_mask]
    y_pred = pred_sem[valid_mask]

    per_class = {}
    f1_list = []
    iou_list = []

    for c in range(1, num_classes + 1):
        class_name = CLASSES[c - 1]
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

        target_count = int(np.sum(y_true == c))
        # Include in macro average only if class is present in target or predicted
        if target_count > 0 or (tp + fp) > 0:
            f1_list.append(f1)
            iou_list.append(iou)

        per_class[class_name] = {
            "category_id": c,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "iou": round(iou, 4),
            "target_pixels": target_count,
            "pred_pixels": int(np.sum(y_pred == c)),
        }

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    macro_iou = float(np.mean(iou_list)) if iou_list else 0.0

    return {
        "macro_f1_ps": round(macro_f1, 4),
        "mIoU_ps": round(macro_iou, 4),
        "utility_C1_ps": round(0.75 * macro_f1 + 0.25 * macro_iou, 4),
        "per_class": per_class,
    }


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_instance_metrics(
    pred_instances: List[Dict[str, Any]],
    target_instances: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Calculate class-aware instance metrics (AP_50^ps, mAP_50:95^ps) via one-to-one matching."""
    if iou_thresholds is None:
        iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    h, w = image_shape

    # Helper to reconstruct binary mask from instance record
    def get_mask(inst: Dict[str, Any]) -> np.ndarray:
        m = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = inst["bbox"]
        if "crop_mask" in inst and inst["crop_mask"] is not None:
            m[y1:y2, x1:x2] = inst["crop_mask"]
        else:
            m[y1:y2, x1:x2] = True
        return m

    # Group by category_id
    cat_preds: Dict[int, List[Dict[str, Any]]] = {}
    cat_gts: Dict[int, List[Dict[str, Any]]] = {}

    for p in pred_instances:
        cat_preds.setdefault(p["category_id"], []).append(p)
    for g in target_instances:
        cat_gts.setdefault(g["category_id"], []).append(g)

    # Sort predictions by score descending
    for cid in cat_preds:
        cat_preds[cid].sort(key=lambda x: x.get("score", 1.0), reverse=True)

    ap_per_iou = {iou_t: [] for iou_t in iou_thresholds}

    for cid in range(1, NUM_CLASSES + 1):
        preds = cat_preds.get(cid, [])
        gts = cat_gts.get(cid, [])

        if not gts and not preds:
            continue

        gt_masks = [get_mask(g) for g in gts]
        pred_masks = [get_mask(p) for p in preds]

        for iou_t in iou_thresholds:
            matched_gt = set()
            tp = 0
            fp = 0

            for p_idx, p_mask in enumerate(pred_masks):
                best_iou = 0.0
                best_gt = -1
                for g_idx, g_mask in enumerate(gt_masks):
                    if g_idx in matched_gt:
                        continue
                    iou = compute_mask_iou(p_mask, g_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g_idx

                if best_iou >= iou_t and best_gt != -1:
                    tp += 1
                    matched_gt.add(best_gt)
                else:
                    fp += 1

            fn = len(gts) - len(matched_gt)
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            ap = p * r  # simplified single-point estimate per class
            ap_per_iou[iou_t].append(ap)

    ap_50 = float(np.mean(ap_per_iou[0.50])) if ap_per_iou[0.50] else 0.0
    all_maps = [float(np.mean(ap_per_iou[t])) for t in iou_thresholds if ap_per_iou[t]]
    map_50_95 = float(np.mean(all_maps)) if all_maps else 0.0

    return {
        "AP50_ps": round(ap_50, 4),
        "mAP50_95_ps": round(map_50_95, 4),
        "utility_C2_ps": round(0.75 * ap_50 + 0.25 * map_50_95, 4),
    }
