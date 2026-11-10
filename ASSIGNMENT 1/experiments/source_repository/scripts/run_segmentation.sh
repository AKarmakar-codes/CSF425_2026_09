#!/usr/bin/env bash
# Standalone shell runner for PCB Component Instance Segmentation
set -e

# Detect python executable (prefer active python/conda env)
PYTHON_CMD="${PYTHON:-python3}"
if ! "$PYTHON_CMD" -c "import torch" &> /dev/null; then
    if command -v python &> /dev/null && python -c "import torch" &> /dev/null; then
        PYTHON_CMD="python"
    elif [ -f "/home/agnibh/anaconda3/envs/myenv/bin/python3" ]; then
        PYTHON_CMD="/home/agnibh/anaconda3/envs/myenv/bin/python3"
    fi
fi

# If arguments are passed with flags (e.g. --image ...) or help
if [[ "$*" == *"--"* ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    exec "$PYTHON_CMD" inference_segmentation.py "$@"
fi

# Positional arguments: $1 = image, $2 = checkpoint, $3 = output
IMAGE_PATH="${1:-data/sample.png}"
CHECKPOINT_PATH="${2:-checkpoints/best_segment.ckpt}"
OUTPUT_PATH="${3:-results/segmentation/prediction.png}"

echo "=== Running PCB Component Instance Segmentation ==="
echo "Input Image: $IMAGE_PATH"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Output:      $OUTPUT_PATH"

exec "$PYTHON_CMD" inference_segmentation.py \
  --image "$IMAGE_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output "$OUTPUT_PATH" "${@:4}"

