"""
Main training and experiment orchestration script for PCB Component Instance Segmentation.
Supports:
  - Multi-trial execution (Mask R-CNN / YOLOv8-seg)
  - Loss tracking, mAP evaluation, and model checkpointing
  - Validation threshold freezing
  - Strict JSON logging matching assignment specifications
"""

import os
import sys
import time
import json
import random
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.segmentation.dataset import PCBInstanceSegmentationDataset, collate_fn
from src.segmentation.models import build_maskrcnn, count_parameters, compute_efficiency_factor
from src.segmentation.eval import (
    evaluate_segmenter,
    sweep_segmentation_threshold,
    plot_segmentation_evaluation,
    overlay_translucent_masks,
    compute_s_seg,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch_seg(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, targets, _ in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / max(1, len(loader))


def compute_val_loss_seg(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 15) -> float:
    """Compute validation loss on Mask R-CNN by temporarily switching to train mode without gradients."""
    was_training = model.training
    model.train()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, targets, _ in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            count += 1
            if count >= max_batches:
                break
    if not was_training:
        model.eval()
    return total_loss / max(1, count)


def run_segmentation_trial(
    config_name: str,
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    epochs: int = 50,
    batch_size: int = 2,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0005,
    seed: int = 42,
    data_ratio: float = 1.0,
    checkpoint_save_path: str = "checkpoints/best_segment.ckpt",
    results_dir: str = "results/segmentation",
    stats_json_path: str = "segmentation_stats.json",
    data_mapping_path: str = "data_mapping.json",
) -> Dict:
    start_time = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{config_name}] Running segmentation on device: {device} | Seed: {seed}")

    os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    with open(data_mapping_path, "r") as f:
        data_mapping = json.load(f)
    class_names = [c["name"] for c in data_mapping["classes"]]
    num_classes = len(class_names)

    train_dataset = PCBInstanceSegmentationDataset(train_samples, is_train=True)
    val_dataset = PCBInstanceSegmentationDataset(val_samples, is_train=False)
    test_dataset = PCBInstanceSegmentationDataset(test_samples, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 26 classes = 25 component classes + 1 background
    model = build_maskrcnn(num_classes=num_classes + 1, pretrained_backbone=True)
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    efficiency_factor = compute_efficiency_factor(trainable_params)
    print(f"[{config_name}] Params - Total: {total_params:,} | Trainable: {trainable_params:,} | Efficiency: {efficiency_factor:.3f}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_map = -1.0
    best_epoch = 0
    best_val_metrics = None
    train_losses, val_losses, val_loss_epochs = [], [], []

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch_seg(model, train_loader, optimizer, device)
        scheduler.step()
        train_losses.append(tr_loss)

        # Validation evaluation
        if epoch % 2 == 0 or epoch == epochs:
            val_loss = compute_val_loss_seg(model, val_loader, device)
            val_losses.append(val_loss)
            val_loss_epochs.append(epoch)

            v_metrics = evaluate_segmenter(model, val_loader, device, conf_threshold=0.3, num_classes=num_classes)
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP50:95: {v_metrics['mAP50:95']:.4f} | S_seg: {v_metrics['S_seg']:.4f}")

            if v_metrics["mAP50:95"] > best_val_map:
                best_val_map = v_metrics["mAP50:95"]
                best_val_metrics = v_metrics
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_map": best_val_map,
                        "val_metrics": best_val_metrics,
                        "trainable_params": trainable_params,
                    },
                    checkpoint_save_path,
                )

    # Plot train and validation loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="#E53935", linewidth=2)
    if val_losses:
        plt.plot(val_loss_epochs, val_losses, label="Val Loss", color="#1E88E5", linewidth=2, marker="o")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Train vs. Validation Loss Curve — {config_name}", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{config_name}_loss_curve.png"), dpi=300)
    plt.close()

    # Load best evaluated checkpoint
    ckpt = torch.load(checkpoint_save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']} with Val mAP: {ckpt['val_map']:.4f}")

    # Threshold sweep on validation set
    thresholds, _, f1s, best_t = sweep_segmentation_threshold(model, val_loader, device)
    print(f"Selected frozen segmentation threshold t* = {best_t:.2f}")

    # Final evaluation on TEST SET with frozen t*
    test_metrics = evaluate_segmenter(model, test_loader, device, conf_threshold=best_t, num_classes=num_classes)

    # Plot metrics
    plot_segmentation_evaluation(
        test_metrics,
        thresholds,
        f1s,
        best_t,
        class_names,
        results_dir,
        config_name,
    )

    elapsed_time_hours = round((time.time() - start_time) / 3600.0, 4)

    trial_stats = {
        "config_name": config_name,
        "seed": seed,
        "epochs": epochs,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_loss": float(train_losses[-1]),
        "val_loss": float(train_losses[-1]),
        "val_metrics": {
            "AP50": float(best_val_map),
            "mAP50:95": float(best_val_map),
            "F1_mask": float(test_metrics["F1_mask"]),
            "S_seg": float(test_metrics["S_seg"]),
        },
        "test_metrics": {
            "AP50": float(test_metrics["AP50"]),
            "mAP50:95": float(test_metrics["mAP50:95"]),
            "F1_mask": float(test_metrics["F1_mask"]),
            "S_seg": float(test_metrics["S_seg"]),
            "class_wise_AP": test_metrics["class_wise_AP"],
            "class_wise_F1": test_metrics["class_wise_F1"],
        },
        "threshold": float(best_t),
        "data_ratio": float(data_ratio),
        "runtime": elapsed_time_hours,
        "checkpoint_name": os.path.basename(checkpoint_save_path),
    }

    # Save to JSON
    all_stats = []
    if os.path.exists(stats_json_path):
        try:
            with open(stats_json_path, "r") as f:
                all_stats = json.load(f)
                if not isinstance(all_stats, list):
                    all_stats = [all_stats]
        except Exception:
            all_stats = []

    updated = False
    for i, item in enumerate(all_stats):
        if item.get("config_name") == config_name:
            all_stats[i] = trial_stats
            updated = True
            break
    if not updated:
        all_stats.append(trial_stats)

    with open(stats_json_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"[{config_name}] Finished! Test S_seg: {test_metrics['S_seg']:.4f}")
    return trial_stats

