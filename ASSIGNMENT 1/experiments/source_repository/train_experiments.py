"""
Master Experiment Runner for CS F425 Deep Learning Project Assignment I.
Orchestrates all required trials for:
  - Exercise 1: PCB Component Classification (3+ trials)
  - Exercise 2: PCB Component Instance Segmentation (3+ trials)
Automatically executes training, validation threshold freezing, test evaluation,
metric logging into classification_stats.json and segmentation_stats.json,
and selects the best overall checkpoints for final submission.

Usage:
  python train_experiments.py --task classification
  python train_experiments.py --task segmentation
  python train_experiments.py --task all
"""

import os
import sys
import json
import argparse
import shutil
import torch

from src.classification.train import run_classification_trial
from src.segmentation.train import run_segmentation_trial
from src.segmentation.models import compute_efficiency_factor as compute_seg_efficiency
from src.classification.models import compute_efficiency_factor as compute_cls_efficiency


def ensure_dataset_links():
    """Ensure data/GraphPCB is symlinked if running on Kaggle/Colab."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/GraphPCB"):
        candidates = [
            "/kaggle/input/the-graphpcb-dataset/GraphPCB",
            "/kaggle/input/the-graphpcb-dataset",
            "/kaggle/input/graphpcb/GraphPCB",
            "/kaggle/input/graphpcb",
            "/kaggle/input/datasets/irislan/graphpcb/GraphPCB",
            "../input/the-graphpcb-dataset/GraphPCB",
            "../input/the-graphpcb-dataset",
            "../input/graphpcb/GraphPCB",
            "../input/graphpcb",
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.isdir(c):
                if os.path.exists(os.path.join(c, "Graph-W")) or os.path.exists(os.path.join(c, "image_dir")):
                    try:
                        os.symlink(os.path.abspath(c), "data/GraphPCB")
                        print(f"🔗 Auto-linked GraphPCB dataset: {c} -> data/GraphPCB")
                        return
                    except Exception:
                        pass


def train_classification_experiments(epochs_override=None, trial_override=None):
    print("=" * 70)
    print("STARTING EXERCISE 1: CLASSIFICATION TRIALS")
    print("=" * 70)

    ensure_dataset_links()
    manifest_path = "data/dataset_manifest.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run python src/data_prep.py first.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    trials = [
        {
            "name": "trial_1_efficientnet_b0",
            "model": "efficientnet_b0",
            "epochs": epochs_override or 40,
            "batch_size": 64,
            "lr": 3e-4,
        },
        {
            "name": "trial_2_mobilenet_v3_large",
            "model": "mobilenet_v3_large",
            "epochs": epochs_override or 40,
            "batch_size": 64,
            "lr": 3e-4,
        },
        {
            "name": "trial_3_convnext_tiny",
            "model": "convnext_tiny",
            "epochs": epochs_override or 15,
            "batch_size": 32,
            "lr": 1e-4,
            "loss_type": "focal",
        },
        {
            "name": "trial_4_efficientnet_b2",
            "model": "efficientnet_b2",
            "epochs": epochs_override or 15,
            "batch_size": 32,
            "lr": 2e-4,
            "loss_type": "class_balanced",
        },
    ]

    if trial_override:
        trials = [t for t in trials if trial_override in t["name"] or trial_override in t["model"]]
        if not trials:
            raise ValueError(f"No trial matches '{trial_override}'. Choose from: trial_1_efficientnet_b0, trial_2_mobilenet_v3_large, trial_3_convnext_tiny, trial_4_efficientnet_b2")

    # Safeguard: check incumbent best score from classification_stats.json
    incumbent_score = -1.0
    if os.path.exists("classification_stats.json"):
        try:
            with open("classification_stats.json", "r") as f:
                c_stats = json.load(f)
            if isinstance(c_stats, list):
                incumbent_score = max(
                    (item.get("val_metrics", {}).get("S_cls", -1.0) for item in c_stats),
                    default=-1.0
                )
        except Exception as e:
            print(f"Notice: Could not read incumbent classification score: {e}")

    best_score = incumbent_score if trial_override else -1.0
    best_ckpt = None

    for t in trials:
        print(f"\n>>> Running {t['name']} ({t['model']}) for {t['epochs']} epochs...")
        ckpt_path = f"checkpoints/{t['name']}.ckpt"
        stats = run_classification_trial(
            config_name=t["name"],
            model_name=t["model"],
            train_samples=manifest["train_crops"],
            val_samples=manifest["val_crops"],
            test_samples=manifest["test_crops"],
            epochs=t["epochs"],
            batch_size=t["batch_size"],
            learning_rate=t["lr"],
            loss_type=t.get("loss_type", "focal"),
            checkpoint_save_path=ckpt_path,
        )

        # Score based on S_cls
        s_score = stats["val_metrics"]["S_cls"]
        e_factor = compute_cls_efficiency(stats["trainable_params"])
        adj_score = s_score * e_factor
        print(f">>> {t['name']} Val S_cls: {s_score:.4f} | E_i: {e_factor:.3f} | Utility: {adj_score:.4f}")

        if s_score > best_score:
            best_score = s_score
            best_ckpt = ckpt_path

    if best_ckpt and os.path.exists(best_ckpt):
        target_path = "checkpoints/best_classification.ckpt"
        shutil.copyfile(best_ckpt, target_path)
        print(f"\nSelected {best_ckpt} as best classification checkpoint (S_cls: {best_score:.4f}) -> copied to {target_path}")
    else:
        print(f"\nIncumbent checkpoint retained (new trial did not exceed incumbent S_cls: {incumbent_score:.4f}).")


def train_segmentation_experiments(epochs_override=None, trial_override=None):
    print("=" * 70)
    print("STARTING EXERCISE 2: INSTANCE SEGMENTATION TRIALS")
    print("=" * 70)

    ensure_dataset_links()

    # Check for RTMDet trial requested
    if trial_override and "rtmdet" in trial_override:
        print("\n>>> Running trial_4_rtmdet_ins_tiny (RTMDet-Ins-tiny)...")
        from src.segmentation.rtmdet_train import run_rtmdet_trial
        stats = run_rtmdet_trial(
            config_name="trial_4_rtmdet_ins_tiny",
            epochs=epochs_override or 60,
            batch_size=8,
            checkpoint_save_path="checkpoints/trial_4_rtmdet_ins_tiny.pth",
        )
        e_factor = compute_seg_efficiency(stats["trainable_params"])
        adj_score = stats["val_metrics"]["S_seg"] * e_factor
        print(f">>> trial_4_rtmdet_ins_tiny Val S_seg: {stats['val_metrics']['S_seg']:.4f} | E_i: {e_factor:.3f} | Adjusted: {adj_score:.4f}")
        return

    manifest_path = "data/dataset_manifest.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run python src/data_prep.py first.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Trial 1 & 2: YOLOv8-seg (if ultralytics is installed)
    yolo_available = False
    try:
        import ultralytics
        from src.segmentation.yolo_train import run_yolo_segmentation_trial
        yolo_available = True
    except ImportError:
        print("Note: 'ultralytics' not found in current environment. To run YOLO trials on HPC: pip install ultralytics")

    # Safeguard: check incumbent best score from segmentation_stats.json
    incumbent_score = -1.0
    if os.path.exists("segmentation_stats.json"):
        try:
            with open("segmentation_stats.json", "r") as f:
                s_stats = json.load(f)
            if isinstance(s_stats, list):
                incumbent_score = max(
                    (item.get("val_metrics", {}).get("S_seg", -1.0) for item in s_stats),
                    default=-1.0
                )
        except Exception as e:
            print(f"Notice: Could not read incumbent segmentation score: {e}")

    best_score = incumbent_score if trial_override else -1.0
    best_ckpt = None

    if yolo_available:
        yolo_trials = [
            {
                "name": "trial_1_yolov8s_seg",
                "variant": "yolov8s-seg.pt",
                "epochs": epochs_override or 60,
                "batch_size": 8,
            },
            {
                "name": "trial_2_yolov8n_seg",
                "variant": "yolov8n-seg.pt",
                "epochs": epochs_override or 60,
                "batch_size": 16,
            },
            {
                "name": "trial_3_yolov8m_seg",
                "variant": "yolov8m-seg.pt",
                "epochs": epochs_override or 50,
                "batch_size": 8,
            },
        ]
        if trial_override:
            yolo_trials = [t for t in yolo_trials if trial_override in t["name"] or trial_override in t["variant"]]

        for t in yolo_trials:
            print(f"\n>>> Running {t['name']} ({t['variant']})...")
            ckpt_path = f"checkpoints/{t['name']}.pt"
            stats = run_yolo_segmentation_trial(
                config_name=t["name"],
                model_variant=t["variant"],
                epochs=t["epochs"],
                batch_size=t["batch_size"],
                checkpoint_save_path=ckpt_path,
            )
            s_score = stats["val_metrics"]["S_seg"]
            e_factor = compute_seg_efficiency(stats["trainable_params"])
            adj_score = s_score * e_factor
            print(f">>> {t['name']} Val S_seg: {s_score:.4f} | E_i: {e_factor:.3f} | Utility: {adj_score:.4f}")
            if s_score > best_score:
                best_score = s_score
                best_ckpt = ckpt_path

    if best_ckpt and os.path.exists(best_ckpt):
        target_path = "checkpoints/best_segment.ckpt"
        shutil.copyfile(best_ckpt, target_path)
        print(f"\nSelected {best_ckpt} as best segmentation checkpoint (S_seg: {best_score:.4f}) -> copied to {target_path}")
    else:
        print(f"\nIncumbent checkpoint retained (new trial did not exceed incumbent S_seg: {incumbent_score:.4f}).")


def main():
    parser = argparse.ArgumentParser(description="Master Training Orchestrator for DL Assignment I")
    parser.add_argument("--task", type=str, choices=["classification", "segmentation", "all"], default="all")
    parser.add_argument("--trial", type=str, default=None, help="Specific trial to run (e.g. trial_3_convnext_tiny, trial_4_rtmdet_ins_tiny)")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch count override for rapid testing")
    args = parser.parse_args()

    if args.task in ["classification", "all"]:
        train_classification_experiments(epochs_override=args.epochs, trial_override=args.trial)
    if args.task in ["segmentation", "all"]:
        train_segmentation_experiments(epochs_override=args.epochs, trial_override=args.trial)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("Generated checkpoints in: checkpoints/")
    print("Generated plots in: results/classification/ and results/segmentation/")
    print("Generated statistics: classification_stats.json and segmentation_stats.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

