"""Export and Finalize Step 4 Winner Checkpoints for Assignment II.

Conforms strictly to Assignment II Master Execution Plan (Step 4):
- Selects winning model (Run 3 Full Re-Engineered Mask2Former).
- Exports lightweight inference-only checkpoints:
    * checkpoints/best_semantic.ckpt (for Q1 semantic segmentation)
    * checkpoints/best_instance.ckpt (for Q2 instance segmentation)
- Audits parameter ceiling and efficiency multiplier E_i = 1.10x.
- Verifies clean-room loading in inference_semantic.py and inference_instance.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_semantic import CompactMask2Former as SemanticModel
from inference_instance import CompactMask2Former as InstanceModel


def export_winner(
    checkpoints_dir: Path = REPO_ROOT / "checkpoints",
    source_ckpt_name: str = "best_run3_full.ckpt",
):
    print("=" * 65)
    print(" CSF425 Assignment II — Finalizing Step 4 (Model Winner Selection)")
    print("=" * 65)

    source_path = checkpoints_dir / source_ckpt_name
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found at: {source_path}")

    print(f"\n1. Loading Winner Checkpoint: {source_path.name}")
    raw_ckpt = torch.load(source_path, map_location="cpu")

    epoch = raw_ckpt.get("epoch", "N/A")
    utility = raw_ckpt.get("best_utility_C1", "N/A")
    val_metrics = raw_ckpt.get("val_metrics", {})
    state_dict = raw_ckpt["state_dict"]

    print(f"   - Selected Model: Run 3 Full Re-Engineered Mask2Former")
    print(f"   - Winner Epoch:   {epoch}")
    print(f"   - Peak Utility:   C1 = {utility}")
    print(f"   - Val Metrics:    {val_metrics}")

    # Build clean inference dictionary (strips 140MB optimizer buffers)
    inf_ckpt = {
        "architecture": "compact_mask2former_r18",
        "epoch": epoch,
        "run_id": 3,
        "run_name": "run3_full",
        "num_classes": 25,
        "num_queries": raw_ckpt.get("num_queries", 150),
        "hidden_dim": 256,
        "num_layers": 6,
        "use_halo": True,
        "params_inference": raw_ckpt.get("params_inference", 18744410),
        "state_dict": state_dict,
        "val_metrics": val_metrics,
        "best_utility_C1": utility,
    }

    # 2. Export semantic checkpoint
    sem_path = checkpoints_dir / "best_semantic.ckpt"
    torch.save(inf_ckpt, sem_path)
    sem_sz = os.path.getsize(sem_path) / (1024 * 1024)
    print(f"\n2. Exported: {sem_path.name} ({sem_sz:.2f} MB)")

    # 3. Export instance checkpoint
    inst_path = checkpoints_dir / "best_instance.ckpt"
    torch.save(inf_ckpt, inst_path)
    inst_sz = os.path.getsize(inst_path) / (1024 * 1024)
    print(f"   Exported: {inst_path.name} ({inst_sz:.2f} MB)")

    # 4. Verify Parameter Count & Bonus Cap
    p_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor) and p.is_floating_point())
    p_bonus_cap = 19504000
    p_hard_cap = 23600000
    e_i = 1.10 if p_count <= p_bonus_cap else 1.0

    print(f"\n3. Parameter Cap & Efficiency Bonus Audit:")
    print(f"   - Total Parameters:     {p_count:,}")
    print(f"   - Efficiency Cap:       {p_bonus_cap:,}")
    print(f"   - Hard Ceiling:         {p_hard_cap:,}")
    print(f"   - Efficiency Bonus Ei:  {e_i:.3f}x (MAXIMUM 1.10x BONUS LOCKED!)")
    assert p_count <= p_bonus_cap, f"Exceeded bonus cap: {p_count} > {p_bonus_cap}"

    # 5. Dual Verification in Standalone Launchers
    print(f"\n4. Verifying Standalone Model Loading:")
    m1 = SemanticModel(num_classes=25, num_queries=150, hidden_dim=256, num_layers=6, use_halo=True)
    m1.load_state_dict(state_dict)
    print("   [PASS] Question 1: CompactMask2Former loads cleanly into inference_semantic.py")

    m2 = InstanceModel(num_classes=25, num_queries=150, hidden_dim=256, num_layers=6, use_halo=True)
    m2.load_state_dict(state_dict)
    print("   [PASS] Question 2: CompactMask2Former loads cleanly into inference_instance.py")

    print("\n" + "=" * 65)
    print(" STEP 4 STATUS: COMPLETE & PASS")
    print(f" Primary Checkpoints Ready in: {checkpoints_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Step 4 Winner Checkpoint")
    parser.add_argument("--ckpt", default="best_run3_full.ckpt", help="Source trained checkpoint filename")
    args = parser.parse_args()

    export_winner(source_ckpt_name=args.ckpt)
