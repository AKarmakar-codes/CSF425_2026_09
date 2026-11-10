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
from src.classification.policy import compute_metrics_dict, compute_s_cls, sweep_threshold, predictions, evaluate_probabilities


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
    all_preds, _ = predictions(all_probs, 0.0)

    if criterion is not None:
        avg_loss = total_loss / max(1, total_samples)
        return all_probs, all_preds, all_targets, avg_loss

    return all_probs, all_preds, all_targets


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
    cm = np.asarray(cm)
    if cm.shape not in [(25,25),(25,26)]:
        raise ValueError("Expected 25x25 or 25x26 confusion matrix")
    plt.figure(figsize=(16 if cm.shape[1] == 26 else 14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{i+1}" for i in range(25)] + (["Reject"] if cm.shape[1] == 26 else []),
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

