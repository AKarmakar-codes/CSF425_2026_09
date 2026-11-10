"""
Main training and experiment orchestration script for PCB Component Classification.
Supports:
  - Multi-trial execution (EfficientNet, MobileNet, ConvNeXt)
  - Validation threshold sweeping & freezing
  - Test set evaluation and full metric logging
  - Strict JSON schema logging matching assignment specifications
"""

import os
import sys
import time
import argparse
import json
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.classification.dataset import (
    PCBComponentCropDataset,
    get_transforms,
    compute_class_weights,
    create_weighted_sampler,
)
from src.classification.models import (
    build_classifier,
    count_parameters,
    compute_efficiency_factor,
)
from src.classification.loss import FocalLoss
from src.classification.eval import (
    evaluate_model,
    compute_metrics_dict,
    sweep_threshold,
    plot_threshold_curves,
    plot_confusion_matrix_heatmap,
    compute_s_cls,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    desc: str = "Training",
) -> float:
    model.train()
    total_loss = 0.0
    use_cuda = (device.type == "cuda")
    for images, targets, _ in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=use_cuda)
        targets = targets.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None and use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def run_classification_trial(
    config_name: str,
    model_name: str = "efficientnet_b0",
    train_samples: List[Dict] = None,
    val_samples: List[Dict] = None,
    test_samples: List[Dict] = None,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    seed: int = 42,
    data_ratio: float = 1.0,
    loss_type: str = "focal",
    checkpoint_save_path: str = "checkpoints/best_classification.ckpt",
    results_dir: str = "results/classification",
    stats_json_path: str = "classification_stats.json",
    data_mapping_path: str = "data_mapping.json",
) -> Dict:
    start_time = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{config_name}] Running on device: {device} | Seed: {seed} | Loss: {loss_type}")

    os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load data mapping
    with open(data_mapping_path, "r") as f:
        data_mapping = json.load(f)
    class_names = [c["name"] for c in data_mapping["classes"]]
    num_classes = len(class_names)

    # 2. Datasets & Loaders
    train_transform = get_transforms(img_size=224, is_train=True)
    eval_transform = get_transforms(img_size=224, is_train=False)

    train_dataset = PCBComponentCropDataset(train_samples, transform=train_transform)
    val_dataset = PCBComponentCropDataset(val_samples, transform=eval_transform)
    test_dataset = PCBComponentCropDataset(test_samples, transform=eval_transform)

    # Balanced sampler for training
    train_sampler = create_weighted_sampler(train_samples, num_classes=num_classes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Model setup
    model = build_classifier(model_name=model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    efficiency_factor = compute_efficiency_factor(trainable_params)
    print(f"[{config_name}] Params - Total: {total_params:,} | Trainable: {trainable_params:,} | Efficiency: {efficiency_factor:.3f}")

    # 4. Loss & Optimizer
    if loss_type == "class_balanced":
        from src.classification.loss import compute_class_balanced_weights
        cb_weights = compute_class_balanced_weights(train_samples, num_classes=num_classes, beta=0.999).to(device)
        criterion = FocalLoss(alpha=cb_weights, gamma=2.0)
        print(f"[{config_name}] Initialized Class-Balanced Focal Loss (beta=0.999, gamma=2.0)")
    elif loss_type == "ce":
        class_weights = compute_class_weights(train_samples, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[{config_name}] Initialized Weighted Cross-Entropy Loss")
    else:
        class_weights = compute_class_weights(train_samples, num_classes=num_classes).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"[{config_name}] Initialized Inverse-Frequency Focal Loss (gamma=2.0)")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 5. Training loop
    best_val_f1 = -1.0
    best_epoch = 0
    best_val_metrics = None  # Store best checkpoint's val metrics
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            desc=f"[{config_name}] Ep {epoch:03d}/{epochs:03d}"
        )
        scheduler.step()

        # Validation step with exact loss calculation
        val_probs, val_preds, val_targets, val_loss = evaluate_model(
            model, val_loader, device, criterion=criterion
        )
        v_metrics = compute_metrics_dict(val_targets, val_preds, num_classes=num_classes)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if v_metrics["F1_macro"] > best_val_f1:
            best_val_f1 = v_metrics["F1_macro"]
            best_val_metrics = v_metrics  # Save best epoch's metrics
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                    "model_name": model_name,
                    "trainable_params": trainable_params,
                },
                checkpoint_save_path,
            )

        print(f"[{config_name}] Ep {epoch:03d}/{epochs:03d} | Tr Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {v_metrics['F1_macro']:.4f} (Best: {best_val_f1:.4f} @ Ep {best_epoch})")

    # 6. Load best checkpoint for validation threshold sweep
    ckpt = torch.load(checkpoint_save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']} with Val F1: {ckpt['val_f1']:.4f}")

    # Sweep threshold t in [0, 1] on VALIDATION SET ONLY
    val_probs, _, val_targets = evaluate_model(model, val_loader, device)
    t_vals, prec_t, rec_t, f1_t, best_t = sweep_threshold(val_probs, val_targets, num_classes=num_classes)
    print(f"Selected frozen threshold t* = {best_t:.2f} on validation set")

    # Persist frozen threshold directly into checkpoint
    ckpt["threshold"] = float(best_t)
    torch.save(ckpt, checkpoint_save_path)

    # Plot P(t), R(t), F1(t)
    threshold_plot_path = os.path.join(results_dir, f"{config_name}_threshold_sweep.png")
    plot_threshold_curves(t_vals, prec_t, rec_t, f1_t, best_t, threshold_plot_path)

    # Plot train and validation loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="#E53935", linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="#1E88E5", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Train vs. Validation Loss Curve — {config_name}", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{config_name}_loss_curve.png"), dpi=300)
    plt.close()

    # 7. Evaluate on official TEST SET
    test_probs, test_preds, test_targets = evaluate_model(model, test_loader, device)
    # Standard closed-set 25-class argmax evaluation
    final_test_preds = np.argmax(test_probs, axis=-1)
    test_metrics = compute_metrics_dict(test_targets, final_test_preds, num_classes=num_classes)

    # Plot test confusion matrix
    cm_plot_path = os.path.join(results_dir, f"{config_name}_confusion_matrix.png")
    plot_confusion_matrix_heatmap(
        np.array(test_metrics["confusion_matrix"]),
        class_names,
        cm_plot_path,
    )

    elapsed_time_hours = round((time.time() - start_time) / 3600.0, 4)

    # Use best checkpoint's val metrics (not last epoch's)
    best_vm = best_val_metrics if best_val_metrics is not None else v_metrics

    # 8. Record trial statistics in strict JSON schema
    trial_stats = {
        "config_name": config_name,
        "loss_type": loss_type,
        "seed": seed,
        "epochs": epochs,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_loss": float(train_losses[-1]),
        "val_loss": float(val_losses[-1]),
        "val_metrics": {
            "accuracy": float(best_vm["accuracy"]),
            "P_macro": float(best_vm["P_macro"]),
            "R_macro": float(best_vm["R_macro"]),
            "F1_macro": float(best_vm["F1_macro"]),
            "S_cls": float(best_vm["S_cls"]),
        },
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "P_macro": float(test_metrics["P_macro"]),
            "R_macro": float(test_metrics["R_macro"]),
            "F1_macro": float(test_metrics["F1_macro"]),
            "S_cls": float(test_metrics["S_cls"]),
            "P_per_class": test_metrics["P_per_class"],
            "R_per_class": test_metrics["R_per_class"],
            "F1_per_class": test_metrics["F1_per_class"],
        },
        "threshold": float(best_t),
        "data_ratio": float(data_ratio),
        "runtime": elapsed_time_hours,
        "checkpoint_name": os.path.basename(checkpoint_save_path),
    }

    # Append / update stats.json
    all_stats = []
    if os.path.exists(stats_json_path):
        try:
            with open(stats_json_path, "r") as f:
                all_stats = json.load(f)
                if not isinstance(all_stats, list):
                    all_stats = [all_stats]
        except Exception:
            all_stats = []

    # Update if config_name exists, else append
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

    print(f"[{config_name}] Finished! Test S_cls: {test_metrics['S_cls']:.4f} (F1_macro: {test_metrics['F1_macro']:.4f})")
    return trial_stats

