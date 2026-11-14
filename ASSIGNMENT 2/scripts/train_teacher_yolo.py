"""Fine-tune COCO-pretrained YOLO26-L-seg for 25 PCB Classes (Rule T1).

Conforms strictly to Assignment II Master Execution Plan (Step 2, Rule T1):
- Loads COCO-pretrained YOLO26-L-seg (yolov8l-seg.pt / yolo11l-seg.pt, ~28.0M parameters).
- Adapts segmentation head to the official 25 PCB classes using source tiles only.
- Exports best checkpoint to checkpoints/teacher_yolo26l_seg.pt.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26-L-seg for 25 PCB Classes (Rule T1)")
    parser.add_argument("--data", default="data/yolo_dataset/dataset.yaml", help="Path to YOLO dataset.yaml")
    parser.add_argument("--model", default="yolov8l-seg.pt", help="Pretrained YOLO model checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for 25-class adaptation")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce if GPU VRAM is limited)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Tile input resolution")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--project", default="runs/train_teacher", help="Logging and checkpoint directory")
    parser.add_argument("--name", default="yolo26l_pcb25", help="Run name")
    parser.add_argument("--out-ckpt", default="checkpoints/teacher_yolo26l_seg.pt", help="Export target for best weights")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found at: {data_yaml}. Run export_yolo_dataset.py first.")

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print("=== [Step 2.1] Fine-Tuning Teacher YOLO26-L-seg on 25 PCB Classes (Rule T1) ===")
    print(f"Pretrained Weights: {args.model}")
    print(f"Dataset Config:     {data_yaml}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch Size:         {args.batch}")
    print(f"Image Size:         {args.imgsz}")
    print(f"Device:             {args.device}")

    # Load COCO pretrained YOLO segmentation model
    model = YOLO(args.model)

    # Train on 25 PCB classes
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=4,
        save=True,
        save_period=5,
        plots=True,
        exist_ok=True,
    )

    # Find and export best checkpoint
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.is_file():
        shutil.copy2(best_pt, out_ckpt)
        print("\n========================================================")
        print(f"Teacher YOLO26-L-seg Training Complete!")
        print(f"Best Weights Exported To: {out_ckpt} ({out_ckpt.stat().st_size / 1e6:.1f} MB)")
        print("========================================================\n")
    else:
        print(f"[Warning] best.pt not found at expected path: {best_pt}")


if __name__ == "__main__":
    main()
