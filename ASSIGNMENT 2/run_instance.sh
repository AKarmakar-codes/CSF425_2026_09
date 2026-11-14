#!/usr/bin/env bash
# run_instance.sh — One-command execution for Question 2 Instance Segmentation.
# Conforms to Assignment II Master Execution Plan & Rubric (Section 7 & 8).
#
# Usage:
#   bash run_instance.sh <dslr_image_or_dir> <output_dir> [--checkpoint PATH] [--gradcam] [--device DEV]
# Example:
#   bash run_instance.sh data/sample.png results/instance/ --checkpoint checkpoints/best_instance.ckpt --gradcam

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

IMAGE="${1:?Error: Image path or directory required as first argument}"
OUT_DIR="${2:?Error: Output directory required as second argument}"
shift 2

CHECKPOINT="${ROOT}/checkpoints/best_instance.ckpt"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --gradcam)
            EXTRA_ARGS+=(--gradcam)
            shift
            ;;
        --device)
            EXTRA_ARGS+=(--device "$2")
            shift 2
            ;;
        --threshold)
            EXTRA_ARGS+=(--threshold "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=== Running Instance Segmentation Pipeline ==="
echo " Input:      $IMAGE"
echo " Output:     $OUT_DIR"
echo " Checkpoint: $CHECKPOINT"

"$PYTHON" "$ROOT/inference_instance.py" \
    --checkpoint "$CHECKPOINT" \
    --images "$IMAGE" \
    --out "$OUT_DIR" \
    "${EXTRA_ARGS[@]}"
