"""
Evaluation, metric computation, curve plotting, and translucent mask visualization
for PCB Component Instance Segmentation.
Implements:
  - S_seg = 0.20*AP50 + 0.50*mAP50:95 + 0.30*F1_mask
  - Class-wise AP and class-wise F1 calculation
  - Precision-Recall curve and F1(t) confidence sweep
  - Translucent mask overlay rendering using the exact 25-color palette
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def compute_s_seg(ap50: float, map50_95: float, f1_mask: float) -> float:
    """Assignment official formula: S_seg = 0.20*AP50 + 0.50*mAP50:95 + 0.30*F1_mask."""
    return float(0.20 * ap50 + 0.50 * map50_95 + 0.30 * f1_mask)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two boolean or binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def evaluate_segmenter(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.3,
    num_classes: int = 25,
) -> Dict:
    """
    Run evaluation on dataloader and compute:
      - AP50, mAP@[0.50:0.95]
      - mask F1 (micro & macro)
      - Class-wise AP and class-wise F1
      - S_seg performance index
    """
    model.eval()
    # Use plain Python float keys to avoid np.float64 key mismatch (e.g., dict[0.50] failing with np.float64 key)
    iou_thresholds = [round(float(x), 2) for x in np.linspace(0.50, 0.95, 10)]

    # Class-level accumulator
    class_tp = {c: {iou: 0 for iou in iou_thresholds} for c in range(1, num_classes + 1)}
    class_fp = {c: {iou: 0 for iou in iou_thresholds} for c in range(1, num_classes + 1)}
    class_fn = {c: {iou: 0 for iou in iou_thresholds} for c in range(1, num_classes + 1)}
    class_gt_counts = {c: 0 for c in range(1, num_classes + 1)}

    with torch.no_grad():
        for images, targets, _ in loader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for pred, tgt in zip(predictions, targets):
                p_boxes = pred["boxes"].cpu().numpy()
                p_scores = pred["scores"].cpu().numpy()
                p_labels = pred["labels"].cpu().numpy()
                p_masks = (pred["masks"].squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)

                t_boxes = tgt["boxes"].numpy()
                t_labels = tgt["labels"].numpy()
                t_masks = tgt["masks"].numpy()

                # Filter by confidence
                keep = p_scores >= conf_threshold
                p_boxes, p_scores, p_labels, p_masks = p_boxes[keep], p_scores[keep], p_labels[keep], p_masks[keep]

                for c in range(1, num_classes + 1):
                    gt_idx = np.where(t_labels == c)[0]
                    pr_idx = np.where(p_labels == c)[0]
                    class_gt_counts[c] += len(gt_idx)

                    if len(gt_idx) == 0 and len(pr_idx) == 0:
                        continue

                    # Compute pairwise mask IoUs
                    ious = np.zeros((len(pr_idx), len(gt_idx)), dtype=np.float32)
                    for i, pi in enumerate(pr_idx):
                        for j, gj in enumerate(gt_idx):
                            ious[i, j] = compute_mask_iou(p_masks[pi], t_masks[gj])

                    for iou_th in iou_thresholds:
                        matched_gt = set()
                        tp = 0
                        fp = 0
                        # Greedily match predictions sorted by score
                        sorted_p = np.argsort(-p_scores[pr_idx]) if len(pr_idx) > 0 else []
                        for pi_order in sorted_p:
                            best_match = -1
                            best_iou = -1.0
                            for gj in range(len(gt_idx)):
                                if gj not in matched_gt and ious[pi_order, gj] >= iou_th:
                                    if ious[pi_order, gj] > best_iou:
                                        best_iou = ious[pi_order, gj]
                                        best_match = gj
                            if best_match >= 0:
                                matched_gt.add(best_match)
                                tp += 1
                            else:
                                fp += 1

                        fn = len(gt_idx) - len(matched_gt)
                        class_tp[c][iou_th] += tp
                        class_fp[c][iou_th] += fp
                        class_fn[c][iou_th] += fn

    # Calculate AP per class and mAP
    class_ap50 = {}
    class_map = {}
    class_f1 = {}

    for c in range(1, num_classes + 1):
        # At IoU = 0.50 (use exact float key)
        tp50 = class_tp[c][0.5]
        fp50 = class_fp[c][0.5]
        fn50 = class_fn[c][0.5]
        p50 = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0
        r50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
        f1_50 = (2 * p50 * r50) / (p50 + r50) if (p50 + r50) > 0 else 0.0
        class_ap50[c] = p50
        class_f1[c] = f1_50

        # Across all IoU thresholds
        aps = []
        for iou_th in iou_thresholds:
            tp_th = class_tp[c][iou_th]
            fp_th = class_fp[c][iou_th]
            prec_th = tp_th / (tp_th + fp_th) if (tp_th + fp_th) > 0 else 0.0
            aps.append(prec_th)
        class_map[c] = float(np.mean(aps))

    valid_classes = [c for c in range(1, num_classes + 1) if class_gt_counts[c] > 0]
    if len(valid_classes) == 0:
        valid_classes = list(range(1, num_classes + 1))

    mean_ap50 = float(np.mean([class_ap50[c] for c in valid_classes]))
    mean_map50_95 = float(np.mean([class_map[c] for c in valid_classes]))
    mean_f1_mask = float(np.mean([class_f1[c] for c in valid_classes]))
    s_seg = compute_s_seg(mean_ap50, mean_map50_95, mean_f1_mask)

    return {
        "AP50": mean_ap50,
        "mAP50:95": mean_map50_95,
        "F1_mask": mean_f1_mask,
        "S_seg": s_seg,
        "class_wise_AP": {c: round(class_map[c], 4) for c in range(1, num_classes + 1)},
        "class_wise_F1": {c: round(class_f1[c], 4) for c in range(1, num_classes + 1)},
        "confidence_threshold": conf_threshold,
    }


def sweep_segmentation_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int = 11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sweep confidence thresholds to determine t* on validation data."""
    thresholds = np.linspace(0.1, 0.9, steps)
    precisions = []
    f1s = []
    best_f1 = -1.0
    best_t = 0.5

    for t in thresholds:
        m = evaluate_segmenter(model, loader, device, conf_threshold=float(t))
        precisions.append(m["AP50"])
        f1s.append(m["F1_mask"])
        if m["F1_mask"] > best_f1:
            best_f1 = m["F1_mask"]
            best_t = float(t)

    return thresholds, np.array(precisions), np.array(f1s), best_t


def plot_segmentation_evaluation(
    eval_metrics: Dict,
    thresholds: np.ndarray,
    f1s: np.ndarray,
    best_t: float,
    class_names: List[str],
    results_dir: str,
    config_name: str,
):
    """Generate and save required instance segmentation plots."""
    os.makedirs(results_dir, exist_ok=True)

    # 1. F1(t) curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1s, marker="o", color="#E53935", linewidth=2.5, label="Mask F1(t)")
    plt.axvline(x=best_t, color="#1E88E5", linestyle="--", label=f"Selected t* = {best_t:.2f}")
    plt.xlabel("Confidence Threshold (t)", fontsize=12)
    plt.ylabel("Mask F1 Score", fontsize=12)
    plt.title("Instance Segmentation Mask F1 vs. Confidence Threshold", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{config_name}_f1_threshold.png"), dpi=300)
    plt.close()

    # 2. Class-wise mAP & F1 bar charts
    c_ids = list(range(1, len(class_names) + 1))
    aps = [eval_metrics["class_wise_AP"].get(c, 0.0) for c in c_ids]
    f1_vals = [eval_metrics["class_wise_F1"].get(c, 0.0) for c in c_ids]

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(c_ids))
    width = 0.38
    ax.bar(x - width/2, aps, width, label="mAP@[0.5:0.95]", color="#1E88E5")
    ax.bar(x + width/2, f1_vals, width, label="Mask F1", color="#43A047")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}. {class_names[c-1][:10]}" for c in c_ids], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Instance Segmentation Performance (mAP and F1)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{config_name}_classwise_metrics.png"), dpi=300)
    plt.close()


def overlay_translucent_masks(
    image_np: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_mapping: Dict,
    save_path: str,
    alpha: float = 0.5,
):
    """
    Overlay translucent instance segmentation masks using the exact 25 colors.
    """
    color_map = {}
    symbol_map = {}
    for c in class_mapping.get("classes", []):
        cid = c["id"]
        # OpenCV uses BGR
        r, g, b = c["rgb"]
        color_map[cid] = (b, g, r)
        symbol_map[cid] = c.get("symbol", f"{cid}")

    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) if image_np.shape[-1] == 3 else image_np.copy()
    overlay = img_bgr.copy()

    for mask, cid, score in zip(masks, labels, scores):
        cid = int(cid)
        color = color_map.get(cid, (0, 255, 0))
        symbol = symbol_map.get(cid, str(cid))

        # Binary mask boolean index
        bool_mask = mask.astype(bool)
        overlay[bool_mask] = color

        # Draw contour boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, color, 2)

        # Draw label at centroid
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            label_text = f"{symbol} {score:.2f}"
            cv2.putText(img_bgr, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Alpha blend overlay with original image
    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, blended)

