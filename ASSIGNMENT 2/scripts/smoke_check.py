"""Smoke Check and Strict Parameter Cap Audit (Step 3).

Conforms strictly to Assignment II Master Execution Plan (Step 3) & Rubric:
- Verifies forward pass of ResNet-18 + FPN + Halo-Dilated Masked-Attention Mask2Former.
- Verifies Seesaw Loss and Laplacian Boundary Loss execution.
- Audits total inference parameter count against:
  * Hard parameter ceiling P_max = P(YOLO26m-seg) = 23,600,000.
  * Efficiency bonus ceiling P_bonus = 19,504,000 (locks in E_i = 1.10x).
- Outputs official audit status string.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.mask2former import CompactMask2Former
from src.losses.seesaw_loss import SeesawLoss
from src.losses.boundary_loss import LaplacianBoundaryLoss


def audit_model(
    verify_cap: int = 23600000,
    target_bonus_cap: int = 19504000,
) -> bool:
    print("=== [Step 3] Executing Model Audit & Smoke Verification ===")
    
    # 1. Initialize model
    model = CompactMask2Former(
        num_classes=25,
        num_queries=150,
        hidden_dim=256,
        num_layers=6,
        pretrained_backbone=False,
        use_halo=True,
    )
    model.eval()

    # 2. Parameter Count Audit
    counts = model.count_parameters()
    p_total = counts["params_total"]
    p_inf = counts["params_inference"]
    p_train = counts["params_trainable"]

    # Efficiency multiplier calculation:
    # E_i = min(1.10, 1.0 + 0.10 * (P_max - P_i) / (P_max - P_bonus)) if P_i <= P_max else 0.0
    if p_inf <= target_bonus_cap:
        e_i = 1.10
    elif p_inf <= verify_cap:
        e_i = 1.0 + 0.10 * (verify_cap - p_inf) / (verify_cap - target_bonus_cap)
    else:
        e_i = 0.0

    print(f"\nParameter Audit Results:")
    print(f"- Total Parameters:     {p_total:,}")
    print(f"- Inference Parameters: {p_inf:,}")
    print(f"- Trainable Parameters: {p_train:,}")
    print(f"- Parameter Ceiling:    {verify_cap:,}")
    print(f"- Bonus Target Cap:     {target_bonus_cap:,}")
    print(f"- Efficiency Bonus E_i: {e_i:.3f}x")

    assert p_inf <= verify_cap, f"FAILED: Parameter count {p_inf} exceeds ceiling {verify_cap}!"
    assert p_inf <= target_bonus_cap, f"WARNING: Parameter count {p_inf} exceeds bonus ceiling {target_bonus_cap}!"

    # 3. Smoke Test Forward Pass
    print("\nExecuting forward pass smoke test (Batch: 2, Image: 512x512)...")
    dummy_img = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        outputs = model(dummy_img)

    assert len(outputs) == 6, f"Expected 6 layer outputs, got {len(outputs)}"
    final_layer = outputs[-1]
    Q = model.num_queries
    assert final_layer["pred_logits"].shape == (2, Q, 26), f"Unexpected logits shape: {final_layer['pred_logits'].shape}"
    assert final_layer["pred_masks"].shape == (2, Q, 128, 128), f"Unexpected masks shape: {final_layer['pred_masks'].shape}"
    print("Forward pass successful!")

    # 4. Smoke Test Semantic Prediction
    print("Testing semantic projection...")
    with torch.no_grad():
        sem_map = model.predict_semantic(dummy_img, score_thresh=0.50)
    assert sem_map.shape == (2, 512, 512), f"Unexpected semantic map shape: {sem_map.shape}"
    print("Semantic projection successful!")

    # 5. Smoke Test Losses
    print("Testing Seesaw and Laplacian Boundary Loss functions...")
    seesaw = SeesawLoss(num_classes=25, p=0.8, q=2.0)
    boundary = LaplacianBoundaryLoss()

    dummy_logits = torch.randn(10, 26)
    dummy_targets = torch.tensor([1, 2, 0, 25, 4, 11, 0, 7, 8, 9])
    loss_seesaw = seesaw(dummy_logits, dummy_targets)
    assert not torch.isnan(loss_seesaw), "Seesaw loss produced NaN!"

    dummy_pred_probs = torch.sigmoid(torch.randn(4, 128, 128))
    dummy_target_masks = (torch.rand(4, 128, 128) > 0.5).float()
    loss_bnd = boundary(dummy_pred_probs, dummy_target_masks)
    assert not torch.isnan(loss_bnd), "Boundary loss produced NaN!"
    print("Loss functions successful!")

    print("\n" + "=" * 60)
    print(f"params_total: {p_total:,} | params_inference: {p_inf:,} | E_i: {e_i:.3f} | STATUS: PASS")
    print("=" * 60 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke Check & Parameter Cap Audit")
    parser.add_argument("--verify-cap", type=int, default=23600000, help="Hard ceiling on parameter count")
    parser.add_argument("--target-bonus-cap", type=int, default=19504000, help="Bonus efficiency ceiling")
    args = parser.parse_args()

    success = audit_model(verify_cap=args.verify_cap, target_bonus_cap=args.target_bonus_cap)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
