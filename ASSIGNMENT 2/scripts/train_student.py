"""Student Model Training Loop — All 4 Ablation Runs.

Conforms strictly to Assignment II Master Execution Plan (Step 4):
- Run 0: Ported Assignment-I baseline (YOLOv8s-seg or reference model).
- Run 1: Vanilla Mask2Former (use_halo=False, uniform CE, no boundary loss).
- Run 2: Compact ablation (hidden_dim=128).
- Run 3: Full final model (use_halo=True, Seesaw, Boundary loss).

Outputs JSON stats files: semantic_stats.json, instance_stats.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Enable PyTorch CUDA expandable segments to prevent memory fragmentation on 16GB GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import CLASSES, NUM_CLASSES, IGNORE_ID, decode_source_mask
from src.models.mask2former import CompactMask2Former
from src.models.matcher import HungarianMatcher
from src.models.criterion import SetCriterion
from src.losses.seesaw_loss import SeesawLoss
from src.losses.boundary_loss import LaplacianBoundaryLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PCBTileDataset(Dataset):
    """Loads prepared 1024x1024 tiles with semantic masks and instance maps."""

    # ImageNet normalization (standard for pretrained ResNet-18 backbone)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        tiles_dir: Path,
        split: str = "train",
        max_instances: int = 150,
        mask_scale: int = 4,  # GT masks downsampled to 1/4 to match decoder output
    ):
        self.tiles_dir = Path(tiles_dir)
        self.split = split
        self.max_instances = max_instances
        self.mask_scale = mask_scale

        meta_file = self.tiles_dir / f"{split}.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"Tile metadata not found: {meta_file}")

        with open(meta_file) as f:
            self.records = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        # Load image
        img_path = self.tiles_dir / rec.get("image", rec.get("image_filename", ""))
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - self.MEAN) / self.STD
        image = torch.from_numpy(img_norm.transpose(2, 0, 1))  # (3, H, W)

        # Load semantic mask (uint8: 0=bg, 1..25=class, 255=ignore)
        sem_key = rec.get("semantic_mask", rec.get("mask_filename", ""))
        sem_path = self.tiles_dir / sem_key
        sem_arr = np.array(Image.open(str(sem_path)), dtype=np.uint8)
        # decode_source_mask expects RGB if stored as RGB, but prepare_source saves
        # uint8 class ID directly (not RGB). Load as grayscale.
        if sem_arr.ndim == 3:
            sem_arr = decode_source_mask(sem_arr)
        semantic = torch.from_numpy(sem_arr.astype(np.int16))  # (H, W), values 0..25, 255

        # Load instance map (uint16)
        inst_key = rec.get("instance_map", "")
        inst_arr = None
        if inst_key:
            inst_path = self.tiles_dir / inst_key
            if inst_path.is_file():
                inst_arr = np.array(Image.open(str(inst_path)), dtype=np.uint16)
        if inst_arr is None:
            inst_arr = np.zeros(sem_arr.shape, dtype=np.uint16)
        instance_map = torch.from_numpy(inst_arr)  # (H, W)

        # Build per-instance GT masks at 1/4 scale
        H, W = sem_arr.shape
        H4, W4 = H // self.mask_scale, W // self.mask_scale

        objects = rec.get("objects", [])
        labels_list = []
        masks_list  = []

        for obj in objects[:self.max_instances]:
            cid = obj.get("category_id", 0)
            if not (1 <= cid <= NUM_CLASSES):
                continue
            uid = obj.get("instance_id", 0)
            if uid == 0:
                continue
            # Binary mask at 1/4 scale
            full_mask = (inst_arr == uid).astype(np.float32)
            small = cv2.resize(full_mask, (W4, H4), interpolation=cv2.INTER_NEAREST)
            if small.sum() < 1:
                continue
            labels_list.append(cid)
            masks_list.append(torch.from_numpy(small))

        if labels_list:
            labels = torch.tensor(labels_list, dtype=torch.long)
            masks  = torch.stack(masks_list, dim=0)   # (M, H4, W4)
        else:
            labels = torch.zeros(0, dtype=torch.long)
            masks  = torch.zeros(0, H4, W4)

        return {
            "image":    image,     # (3, H, W)
            "semantic": semantic,  # (H, W) int16
            "labels":   labels,    # (M,)
            "masks":    masks,     # (M, H4, W4)
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate: stack images/semantics, keep targets as list."""
    images   = torch.stack([b["image"] for b in batch], dim=0)
    semantics = torch.stack([b["semantic"] for b in batch], dim=0)
    targets  = [{"labels": b["labels"], "masks": b["masks"]} for b in batch]
    return {"images": images, "semantics": semantics, "targets": targets}


# ---------------------------------------------------------------------------
# Semantic Loss (direct from semantic map — auxiliary to instance matching)
# ---------------------------------------------------------------------------

def semantic_ce_loss(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    semantic_gt: torch.Tensor,
    ignore_index: int = IGNORE_ID,
) -> torch.Tensor:
    """Compute semantic cross-entropy on the class-probability map vs GT semantic mask.

    Uses the weighted query combination formula:
        S_c(h,w) = sum_q P_q(c) * sigmoid(M_q(h,w))
    Then applies cross-entropy against ground-truth semantic labels.
    """
    B = pred_logits.shape[0]
    H, W = semantic_gt.shape[-2:]
    H4, W4 = pred_masks.shape[-2:]

    pred_logits = pred_logits.float()
    pred_masks  = pred_masks.float()

    probs = F.softmax(pred_logits, dim=-1)       # (B, Q, C+1)
    class_probs = probs[:, :, 1:]                # (B, Q, C)   classes 1..25
    masks_up = F.interpolate(
        pred_masks, size=(H, W), mode="bilinear", align_corners=False
    ).sigmoid()                                  # (B, Q, H, W)

    # S_c(h,w) = sum_q P_q(c) * M_q(h,w)   →   (B, C, H, W)
    sem_probs = torch.einsum("bqc,bqhw->bchw", class_probs, masks_up)

    # Background probability: 1 - sum(class_probs * masks_up)
    bg_prob = (1.0 - sem_probs.sum(dim=1, keepdim=True)).clamp(min=1e-4)
    sem_probs_full = torch.cat([bg_prob, sem_probs.clamp(min=1e-4)], dim=1)
    sem_probs_full = sem_probs_full / sem_probs_full.sum(dim=1, keepdim=True)

    gt = semantic_gt.long()
    gt[gt == 255] = -100

    loss = F.nll_loss(sem_probs_full.log(), gt, ignore_index=-100, reduction="mean")
    return loss


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(run_id: int, num_queries: int = 150) -> CompactMask2Former:
    """Build model for the specified ablation run."""
    if run_id == 1:
        # Run 1: Vanilla Mask2Former (no halo dilation)
        return CompactMask2Former(
            num_classes=NUM_CLASSES,
            num_queries=num_queries,
            hidden_dim=256,
            num_layers=6,
            pretrained_backbone=True,
            use_halo=False,
        )
    elif run_id == 2:
        # Run 2: Compact ablation (hidden_dim=128)
        return CompactMask2Former(
            num_classes=NUM_CLASSES,
            num_queries=num_queries,
            hidden_dim=128,
            num_layers=6,
            pretrained_backbone=True,
            use_halo=True,
        )
    else:
        # Run 3 (default) & Run 0 fallback: Full model
        return CompactMask2Former(
            num_classes=NUM_CLASSES,
            num_queries=num_queries,
            hidden_dim=256,
            num_layers=6,
            pretrained_backbone=True,
            use_halo=True,
        )


def build_criterion(run_id: int) -> SetCriterion:
    """Build training criterion for the specified ablation run."""
    if run_id == 1:
        # Run 1: Uniform CE (no Seesaw), no boundary loss
        return SetCriterion(
            num_classes=NUM_CLASSES,
            weight_boundary=0.0,  # Disable boundary loss for vanilla run
        )
    else:
        # Run 2 & 3: Full Seesaw + Boundary criterion
        return SetCriterion(
            num_classes=NUM_CLASSES,
            weight_boundary=1.0,
        )


def build_optimizer(model: nn.Module, lr: float, backbone_lr: float, weight_decay: float):
    """Build AdamW optimizer with differential learning rates (backbone vs decoder)."""
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    return optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": other_params,    "lr": lr},
    ], weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Single Epoch Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: CompactMask2Former,
    criterion: SetCriterion,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[Any] = None,
    use_amp: bool = False,
    grad_clip: float = 1.0,
    epoch: int = 0,
    sem_loss_weight: float = 1.0,
    grad_accum_steps: int = 1,
    print_freq: int = 50,
) -> Dict[str, float]:
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0
    total_steps = len(loader)

    optimizer.zero_grad()
    t_start = time.time()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for step, batch in enumerate(loader):
        images    = batch["images"].to(device, non_blocking=True)
        semantics = batch["semantics"].to(device, non_blocking=True)
        targets   = [
            {
                "labels": t["labels"].to(device),
                "masks":  t["masks"].to(device),
            }
            for t in batch["targets"]
        ]

        if hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast(autocast_device, enabled=use_amp)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

        with autocast_ctx:
            outputs = model(images)   # List[6 dicts]
            loss_dict = criterion(outputs, targets)

            final = outputs[-1]
            loss_sem = semantic_ce_loss(
                final["pred_logits"], final["pred_masks"], semantics
            )
            loss_dict["loss_semantic"] = loss_sem
            loss_total = loss_dict["loss_total"] + sem_loss_weight * loss_sem
            loss_dict["loss_total"] = loss_total
            loss_scaled = loss_total / grad_accum_steps

        if scaler is not None and use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
            if scaler is not None and use_amp:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + float(v.item())
        n_batches += 1

        if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
            elapsed = time.time() - t_start
            print(
                f"  [Epoch {epoch:2d} | Step {step+1:4d}/{total_steps}] "
                f"Loss: {loss_total.item():.4f} "
                f"(cls: {loss_dict['loss_class'].item():.4f}, "
                f"mask: {loss_dict['loss_mask'].item():.4f}, "
                f"dice: {loss_dict['loss_dice'].item():.4f}, "
                f"bnd: {loss_dict['loss_boundary'].item():.4f}, "
                f"sem: {loss_sem.item():.4f}) | {elapsed:.0f}s",
                flush=True
            )

    return {k: v / max(1, n_batches) for k, v in total_losses.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: CompactMask2Former,
    loader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.50,
) -> Dict[str, float]:
    model.eval()
    K = NUM_CLASSES + 1  # 26 classes (0=bg, 1..25 PCB classes)
    conf = np.zeros((K, K), dtype=np.int64)

    for batch in loader:
        images    = batch["images"].to(device, non_blocking=True)
        semantics = batch["semantics"].to(device, non_blocking=True)

        pred_sem = model.predict_semantic(images, score_thresh=score_thresh)  # (B, H, W)

        # Fast vectorized confusion matrix accumulation
        gt_np = semantics.cpu().numpy()
        pr_np = pred_sem.cpu().numpy()
        valid = gt_np != IGNORE_ID
        idx = gt_np[valid] * K + pr_np[valid]
        conf += np.bincount(idx, minlength=K * K).reshape(K, K)

    # Macro F1 and mIoU (classes 1..25 only, background 0 excluded per plan)
    f1_per = []
    iou_per = []
    for c in range(1, NUM_CLASSES + 1):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        if (tp + fp + fn) == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        f1_per.append(f1)
        iou_per.append(iou)

    macro_f1  = float(np.mean(f1_per))  if f1_per  else 0.0
    macro_iou = float(np.mean(iou_per)) if iou_per else 0.0
    utility_c1 = 0.75 * macro_f1 + 0.25 * macro_iou

    return {
        "macro_f1":   round(macro_f1, 4),
        "mIoU":       round(macro_iou, 4),
        "utility_C1": round(utility_c1, 4),
    }


# ---------------------------------------------------------------------------
# Main Training Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Compact Mask2Former Student (All Ablation Runs)")
    parser.add_argument("--tiles-dir",   default="data/prepared_tiles", help="Prepared tiles directory")
    parser.add_argument("--out-dir",     default="checkpoints",          help="Checkpoint output directory")
    parser.add_argument("--run",         type=int, default=3, choices=[1, 2, 3],
                        help="Ablation run: 1=Vanilla, 2=Compact(h=128), 3=Full")
    parser.add_argument("--epochs",      type=int, default=60)
    parser.add_argument("--batch-size",  type=int, default=2,
                        help="Micro batch size per step (default: 2, safely fits in 16GB GPU)")
    parser.add_argument("--effective-batch-size", type=int, default=8,
                        help="Effective batch size via gradient accumulation (plan: 8)")
    parser.add_argument("--lr",          type=float, default=1e-4,  help="Decoder learning rate (AdamW)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,  help="Backbone learning rate (AdamW)")
    parser.add_argument("--weight-decay",type=float, default=0.05,  help="AdamW weight decay (plan: 0.05)")
    parser.add_argument("--grad-clip",   type=float, default=1.0)
    parser.add_argument("--num-queries", type=int,   default=150,   help="Number of object queries (plan: Q=150)")
    parser.add_argument("--amp",         action="store_true", default=True,
                        help="Use Automatic Mixed Precision (AMP fp16) to conserve VRAM (default: True)")
    parser.add_argument("--no-amp",      action="store_false", dest="amp",
                        help="Disable AMP mixed precision")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--workers",     type=int, default=2)
    parser.add_argument("--print-freq",  type=int, default=50,      help="Logging frequency in steps")
    parser.add_argument("--resume",      type=str, default=None,    help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    tiles_dir = Path(args.tiles_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = {1: "run1_vanilla", 2: "run2_compact", 3: "run3_full"}.get(args.run, f"run{args.run}")
    grad_accum_steps = max(1, args.effective_batch_size // args.batch_size)
    effective_bs = args.batch_size * grad_accum_steps

    use_amp = args.amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n=== [Step 4] Training Ablation {run_name.upper()} on {device} ===")
    print(f"Q={args.num_queries} | LR={args.lr} | backbone_LR={args.backbone_lr} | WD={args.weight_decay}")
    print(f"Batch Size (per-step): {args.batch_size} | Accumulation Steps: {grad_accum_steps} | Effective Batch Size: {effective_bs}")
    print(f"AMP Mixed Precision:   {'ENABLED (fp16)' if use_amp else 'DISABLED'}")

    # Datasets & Loaders
    train_ds = PCBTileDataset(tiles_dir, split="train", max_instances=args.num_queries)
    val_ds   = PCBTileDataset(tiles_dir, split="val",   max_instances=args.num_queries)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # Model & Criterion
    model = build_model(args.run, num_queries=args.num_queries).to(device)
    criterion = build_criterion(args.run).to(device)
    optimizer = build_optimizer(model, lr=args.lr, backbone_lr=args.backbone_lr,
                                weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    p_inf = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {p_inf:,} ({'PASS' if p_inf <= 19504000 else 'OVER BONUS CAP'})")

    best_utility = 0.0
    best_ckpt    = out_dir / f"best_{run_name}.ckpt"
    latest_ckpt  = out_dir / f"latest_{run_name}.ckpt"
    history_path = out_dir / f"{run_name}_history.json"
    history      = []
    start_epoch  = 1

    if args.resume == "auto":
        if latest_ckpt.is_file():
            args.resume = str(latest_ckpt)
        elif best_ckpt.is_file():
            args.resume = str(best_ckpt)

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint to resume: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_utility = checkpoint.get("best_utility_C1", 0.0)
        print(f"Resuming at epoch {start_epoch} (previous best utility C1: {best_utility:.4f})")

        # Fast forward scheduler to match resumed epoch
        for _ in range(start_epoch - 1):
            scheduler.step()

        # Load existing history if available
        if history_path.is_file():
            try:
                history = json.loads(history_path.read_text())
                history = [h for h in history if h.get("epoch", 0) < start_epoch]
                print(f"Retained {len(history)} previous epochs in history log.")
            except Exception:
                history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            scaler=scaler, use_amp=use_amp,
            grad_clip=args.grad_clip, epoch=epoch,
            grad_accum_steps=grad_accum_steps, print_freq=args.print_freq,
        )
        scheduler.step()

        val_metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0

        utility = val_metrics["utility_C1"]
        ckpt_data = {
            "epoch": epoch,
            "run_id": args.run,
            "run_name": run_name,
            "num_queries": args.num_queries,
            "params_inference": p_inf,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_utility_C1": max(best_utility, utility),
        }
        # Always save latest checkpoint to allow seamless resumption
        torch.save(ckpt_data, str(latest_ckpt))

        if utility > best_utility:
            best_utility = utility
            torch.save(ckpt_data, str(best_ckpt))

        log_line = {
            "epoch": epoch,
            **{f"train_{k}": round(v, 6) for k, v in train_losses.items()},
            **{f"val_{k}":   round(v, 6) for k, v in val_metrics.items()},
        }
        history.append(log_line)

        print(
            f"[Epoch {epoch:3d}/{args.epochs}] "
            f"loss={train_losses.get('loss_total', 0):.4f} | "
            f"F1={val_metrics['macro_f1']:.4f} mIoU={val_metrics['mIoU']:.4f} "
            f"U1={utility:.4f} | {elapsed:.0f}s"
        )

    # Write training history
    history_path = out_dir / f"{run_name}_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    # Write final stats
    best_ckpt_data = torch.load(str(best_ckpt), map_location="cpu")
    stats = {
        "run_id":          args.run,
        "run_name":        run_name,
        "num_queries":     args.num_queries,
        "hidden_dim":      256 if args.run != 2 else 128,
        "use_halo":        args.run != 1,
        "params_inference": p_inf,
        "best_epoch":      best_ckpt_data["epoch"],
        "best_utility_C1": best_utility,
        "best_macro_f1":   best_ckpt_data["val_metrics"]["macro_f1"],
        "best_mIoU":       best_ckpt_data["val_metrics"]["mIoU"],
        "checkpoint":      str(best_ckpt),
        "history":         str(history_path),
    }
    stats_path = out_dir / f"{run_name}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\n=== Training Complete: {run_name.upper()} ===")
    print(f"Best Utility C1: {best_utility:.4f} @ epoch {best_ckpt_data['epoch']}")
    print(f"Checkpoint: {best_ckpt}")
    print(f"Stats:      {stats_path}")


if __name__ == "__main__":
    main()
