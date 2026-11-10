"""
Evaluation, threshold sweeping, plotting, and visualization for PCB Component Classification.
Implements:
  - Macro Precision, Recall, F1, and S_cls formula: S_cls = 0.20*P_macro + 0.20*R_macro + 0.60*F1_macro
  - P(t), R(t), F1(t) threshold sweeping and t* selection on validation
  - Confusion matrix visualization
  - Full PCB board visualization with colored bounding boxes
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


def compute_s_cls(p_macro: float, r_macro: float, f1_macro: float) -> float:
    """Assignment official formula: S_cls = 0.20*P_macro + 0.20*R_macro + 0.60*F1_macro."""
    return float(0.20 * p_macro + 0.20 * r_macro + 0.60 * f1_macro)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple:
    """
    Evaluate model and extract raw probabilities, predicted classes, targets, and optionally loss.
    If criterion is None: returns (all_probs, all_preds, all_targets)
    If criterion is provided: returns (all_probs, all_preds, all_targets, avg_loss)
    """
    model.eval()
    all_probs = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets_dev = targets.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, targets_dev)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

            probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.argmax(all_probs, axis=-1)

    if criterion is not None:
        avg_loss = total_loss / max(1, total_samples)
        return all_probs, all_preds, all_targets, avg_loss

    return all_probs, all_preds, all_targets


def compute_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 25,
) -> Dict:
    """Compute comprehensive macro and per-class classification metrics."""
    # Strictly pass all canonical labels so macro metrics average across all 25 classes
    class_labels = list(range(num_classes))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )

    acc = accuracy_score(y_true, y_pred)
    s_cls = compute_s_cls(p_macro, r_macro, f1_macro)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "accuracy": float(acc),
        "P_macro": float(p_macro),
        "R_macro": float(r_macro),
        "F1_macro": float(f1_macro),
        "S_cls": float(s_cls),
        "P_per_class": [float(x) for x in p_per_class],
        "R_per_class": [float(x) for x in r_per_class],
        "F1_per_class": [float(x) for x in f1_per_class],
        "support": [int(x) for x in support],
        "confusion_matrix": cm.tolist(),
    }


def sweep_threshold(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    num_classes: int = 25,
    steps: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Sweep confidence thresholds t in [0, 1] on validation probabilities.
    Returns: (t_values, precisions, recalls, f1s, best_t)
    """
    t_values = np.linspace(0.0, 1.0, steps)
    precisions = []
    recalls = []
    f1s = []

    best_f1 = -1.0
    best_t = 0.5

    for t in t_values:
        max_probs = np.max(all_probs, axis=-1)
        preds = np.argmax(all_probs, axis=-1)

        # Samples where max probability < threshold are assigned a pseudo-void label (-1)
        masked_preds = np.where(max_probs >= t, preds, -1)

        # Compute macro metrics ignoring void
        # Count true positives, false positives, false negatives per class
        p_c, r_c, f1_c = [], [], []
        for c in range(num_classes):
            tp = np.sum((masked_preds == c) & (all_targets == c))
            fp = np.sum((masked_preds == c) & (all_targets != c))
            fn = np.sum((masked_preds != c) & (all_targets == c))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            p_c.append(prec)
            r_c.append(rec)
            f1_c.append(f1)

        p_macro = float(np.mean(p_c))
        r_macro = float(np.mean(r_c))
        f1_macro = float(np.mean(f1_c))

        precisions.append(p_macro)
        recalls.append(r_macro)
        f1s.append(f1_macro)

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_t = float(t)

    return (
        t_values,
        np.array(precisions),
        np.array(recalls),
        np.array(f1s),
        best_t,
    )


def plot_threshold_curves(
    t_values: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1s: np.ndarray,
    best_t: float,
    save_path: str,
):
    """Plot P(t), R(t), and F1(t) curves and highlight t*."""
    plt.figure(figsize=(8, 6))
    plt.plot(t_values, precisions, label="Macro Precision P(t)", color="#1E88E5", linewidth=2)
    plt.plot(t_values, recalls, label="Macro Recall R(t)", color="#43A047", linewidth=2)
    plt.plot(t_values, f1s, label="Macro F1(t)", color="#E53935", linewidth=2.5)
    plt.axvline(x=best_t, color="#8E24AA", linestyle="--", label=f"Selected t* = {best_t:.2f}")

    plt.xlabel("Confidence Threshold (t)", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.title("Classification Metric Behavior vs. Confidence Threshold", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{i+1}" for i in range(len(class_names))],
        yticklabels=[f"{i+1}. {name[:12]}" for i, name in enumerate(class_names)],
    )
    plt.xlabel("Predicted Class ID (1-25)", fontsize=12)
    plt.ylabel("Ground Truth Class ID", fontsize=12)
    plt.title("PCB Component Classification Confusion Matrix", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_full_board_with_boxes(
    image_path: str,
    boxes: List[Tuple[int, int, int, int]],
    pred_classes: List[int],
    pred_confs: List[float],
    class_mapping: Dict,
    save_path: str,
):
    """
    Overlay colored bounding boxes on full PCB image using the canonical 25 colors.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Build color lookup by 1-indexed class ID
    color_map = {}
    for cls_info in class_mapping.get("classes", []):
        cid = cls_info["id"]
        rgb = tuple(cls_info["rgb"])
        color_map[cid] = (rgb, cls_info.get("symbol", f"{cid}"))

    for (x1, y1, x2, y2), cid, conf in zip(boxes, pred_classes, pred_confs):
        rgb, symbol = color_map.get(cid, ((255, 255, 255), str(cid)))
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)
        label_text = f"{symbol} {conf:.2f}"
        # Small background rectangle for readability
        draw.rectangle([x1, max(0, y1 - 16), x1 + len(label_text) * 8, y1], fill=rgb)
        draw.text((x1 + 2, max(0, y1 - 15)), label_text, fill=(255, 255, 255))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

