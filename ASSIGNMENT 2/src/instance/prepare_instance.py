"""Instance preparation and geometric post-processing utilities.

Conforms to Assignment II Master Execution Plan (Q2 Instance Segmentation):
- Extracts bounding boxes and binary masks from uint16 instance maps.
- Builds targets for Hungarian matching.
- Performs class-aware Mask-IoU and containment NMS across overlapping tiles.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from src.constants import CLASSES, NUM_CLASSES, BACKGROUND_ID, IGNORE_ID


def extract_instance_bboxes(
    instance_map: np.ndarray,
    semantic_map: Optional[np.ndarray] = None,
    min_area: int = 4,
) -> List[Dict[str, Any]]:
    """Extract bounding boxes and class IDs from a 2D instance map."""
    instances = []
    unique_ids = [uid for uid in np.unique(instance_map) if uid > 0]

    for uid in unique_ids:
        bin_mask = (instance_map == uid)
        area = int(bin_mask.sum())
        if area < min_area:
            continue

        ys, xs = np.where(bin_mask)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max() + 1), int(ys.max() + 1)

        cid = 0
        if semantic_map is not None:
            cids = semantic_map[bin_mask]
            valid_cids = cids[(cids > 0) & (cids <= NUM_CLASSES)]
            if len(valid_cids) > 0:
                cid = int(np.bincount(valid_cids).argmax())

        crop = bin_mask[y1:y2, x1:x2]
        instances.append({
            "instance_id": int(uid),
            "category_id": cid,
            "class_code": CLASSES[cid - 1] if 1 <= cid <= NUM_CLASSES else "UNK",
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "crop_mask": crop,
        })

    return instances


def build_instance_targets(
    instance_map: torch.Tensor,
    semantic_map: torch.Tensor,
    mask_scale: int = 4,
    max_instances: int = 150,
) -> Dict[str, torch.Tensor]:
    """Build downscaled ground-truth targets (labels and binary masks) for Hungarian matching."""
    inst_np = instance_map.cpu().numpy()
    sem_np = semantic_map.cpu().numpy()
    h, w = inst_np.shape
    h4, w4 = h // mask_scale, w // mask_scale

    instances = extract_instance_bboxes(inst_np, sem_np)
    labels = []
    masks = []

    for inst in instances[:max_instances]:
        cid = inst["category_id"]
        if not (1 <= cid <= NUM_CLASSES):
            continue
        uid = inst["instance_id"]
        bin_m = (inst_np == uid).astype(np.float32)
        small_m = cv2.resize(bin_m, (w4, h4), interpolation=cv2.INTER_NEAREST)
        if small_m.sum() < 1:
            continue
        labels.append(cid)
        masks.append(torch.from_numpy(small_m))

    if labels:
        return {
            "labels": torch.tensor(labels, dtype=torch.long, device=instance_map.device),
            "masks": torch.stack(masks, dim=0).to(device=instance_map.device),
        }
    return {
        "labels": torch.zeros(0, dtype=torch.long, device=instance_map.device),
        "masks": torch.zeros(0, h4, w4, device=instance_map.device),
    }


def compute_instance_iou_matrix(
    masks_a: List[np.ndarray],
    masks_b: List[np.ndarray],
) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of binary masks."""
    na, nb = len(masks_a), len(masks_b)
    iou_mat = np.zeros((na, nb), dtype=np.float32)
    if na == 0 or nb == 0:
        return iou_mat

    for i, ma in enumerate(masks_a):
        for j, mb in enumerate(masks_b):
            inter = np.logical_and(ma, mb).sum()
            union = np.logical_or(ma, mb).sum()
            iou_mat[i, j] = float(inter / union) if union > 0 else 0.0
    return iou_mat


def apply_instance_nms(
    candidates: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
    containment_threshold: float = 0.70,
) -> List[Dict[str, Any]]:
    """Apply class-aware Mask-IoU and containment NMS across overlapping tiles."""
    if len(candidates) <= 1:
        return candidates

    candidates.sort(key=lambda x: x.get("score", 1.0), reverse=True)
    kept = []

    for cand in candidates:
        duplicate = False
        c_cid = cand["category_id"]
        cx1, cy1, cx2, cy2 = cand["bbox"]
        c_mask = cand.get("mask")

        for prev in kept:
            if c_cid != prev["category_id"]:
                continue
            px1, py1, px2, py2 = prev["bbox"]
            p_mask = prev.get("mask")

            # Check bbox overlap
            ix1, iy1 = max(cx1, px1), max(cy1, py1)
            ix2, iy2 = min(cx2, px2), min(cy2, py2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            if c_mask is not None and p_mask is not None:
                # Mask-level IoU and containment
                cm_crop = c_mask[iy1 - cy1 : iy2 - cy1, ix1 - cx1 : ix2 - cx1]
                pm_crop = p_mask[iy1 - py1 : iy2 - py1, ix1 - px1 : ix2 - px1]
                inter = np.logical_and(cm_crop, pm_crop).sum()
                c_area = cand.get("area", c_mask.sum())
                p_area = prev.get("area", p_mask.sum())
                union = c_area + p_area - inter
                iou = float(inter / union) if union > 0 else 0.0
                containment = float(inter / min(c_area, p_area)) if min(c_area, p_area) > 0 else 0.0
                if iou > iou_threshold or containment > containment_threshold:
                    duplicate = True
                    break
            else:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                c_area = (cx2 - cx1) * (cy2 - cy1)
                p_area = (px2 - px1) * (py2 - py1)
                union = c_area + p_area - inter_area
                iou = float(inter_area / union) if union > 0 else 0.0
                if iou > iou_threshold:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(cand)

    return kept
