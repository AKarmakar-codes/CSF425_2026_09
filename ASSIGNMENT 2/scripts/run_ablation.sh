#!/usr/bin/env bash
# run_ablation.sh — Train all 3 student model ablation runs (Runs 1, 2, 3)
#
# Usage:
#   bash scripts/run_ablation.sh [--tiles-dir PATH] [--epochs N] [--device DEVICE]
#
# Runs:
#   Run 1: Vanilla Mask2Former (use_halo=False, uniform CE, no boundary loss)
#   Run 2: Compact ablation (hidden_dim=128)
#   Run 3: Full final model  (use_halo=True, Seesaw, Boundary loss)
#
# Note: Run 0 (Assignment-I baseline) is a separate model (see README.md).

set -e

TILES_DIR="${TILES_DIR:-data/prepared_tiles}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-2}"
DEVICE="${DEVICE:-auto}"
NUM_QUERIES="${NUM_QUERIES:-150}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"

# Parse optional CLI overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --tiles-dir)   TILES_DIR="$2";   shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --batch-size)  BATCH="$2";       shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --num-queries) NUM_QUERIES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "============================================================"
echo " CSF425 Assignment II — Student Ablation Training"
echo "============================================================"
echo " Tiles Dir:   $TILES_DIR"
echo " Epochs:      $EPOCHS"
echo " Batch Size:  $BATCH"
echo " Num Queries: $NUM_QUERIES"
echo " Device:      $DEVICE"
echo "============================================================"

COMMON_ARGS=(
    --tiles-dir "$TILES_DIR"
    --epochs    "$EPOCHS"
    --batch-size "$BATCH"
    --device    "$DEVICE"
    --num-queries "$NUM_QUERIES"
    --lr "$LR"
    --backbone-lr "$BACKBONE_LR"
    --weight-decay "$WEIGHT_DECAY"
)

# --- Run 1: Vanilla Mask2Former ---
echo ""
echo "[Run 1/3] Vanilla Mask2Former (use_halo=False, uniform CE, no boundary loss)"
$PYTHON scripts/train_student.py --run 1 "${COMMON_ARGS[@]}" \
    || { echo "[Error] Run 1 failed!"; exit 1; }
echo "[Run 1] Complete."

# --- Run 2: Compact Ablation (hidden_dim=128) ---
echo ""
echo "[Run 2/3] Compact Ablation (hidden_dim=128)"
$PYTHON scripts/train_student.py --run 2 "${COMMON_ARGS[@]}" \
    || { echo "[Error] Run 2 failed!"; exit 1; }
echo "[Run 2] Complete."

# --- Run 3: Full Final Model ---
echo ""
echo "[Run 3/3] Full Final Model (use_halo=True, Seesaw, Boundary)"
$PYTHON scripts/train_student.py --run 3 "${COMMON_ARGS[@]}" \
    || { echo "[Error] Run 3 failed!"; exit 1; }
echo "[Run 3] Complete."

# --- Copy best full model checkpoint ---
if [ -f checkpoints/best_run3_full.ckpt ]; then
    cp checkpoints/best_run3_full.ckpt checkpoints/best_semantic.ckpt
    cp checkpoints/best_run3_full.ckpt checkpoints/best_instance.ckpt
    cp checkpoints/best_run3_full.ckpt checkpoints/best_segment.ckpt
    echo "Copied Run 3 (best_run3_full.ckpt) → best_semantic.ckpt, best_instance.ckpt, best_segment.ckpt"
fi

# --- Aggregate stats into semantic_stats.json and instance_stats.json ---
$PYTHON - <<'PYEOF'
import json
from pathlib import Path

ckpt = Path("checkpoints")
runs = []
for run_id in [1, 2, 3]:
    name = {1: "run1_vanilla", 2: "run2_compact", 3: "run3_full"}[run_id]
    sf = ckpt / f"{name}_stats.json"
    if sf.is_file():
        runs.append(json.loads(sf.read_text()))

semantic_stats = {
    "metric": "macro_F1^ps and mIoU^ps on source val set (non-ignored pixels)",
    "ignore_index": 255,
    "background_contributes_to_FP": True,
    "runs": [{"run_id": r["run_id"], "run_name": r["run_name"],
               "params_inference": r["params_inference"],
               "use_halo": r["use_halo"], "hidden_dim": r["hidden_dim"],
               "macro_F1": r.get("best_macro_f1", 0.0),
               "mIoU": r.get("best_mIoU", 0.0),
               "utility_C1": r.get("best_utility_C1", 0.0)} for r in runs]
}
instance_stats = {
    "metric": "AP_50^ps and mAP_50:95^ps on source val set",
    "note": "Computed after full training via diagnostic_metrics.py",
    "runs": [{"run_id": r["run_id"], "run_name": r["run_name"],
               "params_inference": r["params_inference"]} for r in runs]
}

Path("semantic_stats.json").write_text(json.dumps(semantic_stats, indent=2))
Path("instance_stats.json").write_text(json.dumps(instance_stats, indent=2))
print("Wrote semantic_stats.json and instance_stats.json")
PYEOF

echo ""
echo "============================================================"
echo " Ablation Training Complete!"
echo " Checkpoints saved to: checkpoints/"
echo " Stats:  semantic_stats.json  instance_stats.json"
echo "============================================================"
