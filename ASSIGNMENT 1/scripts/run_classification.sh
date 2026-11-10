#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON_CMD:-python}"
if [[ "${1:-}" == --* ]]; then
  exec "$PYTHON_CMD" inference_classification.py "$@"
fi
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 IMAGE CHECKPOINT OUTPUT BOXES_JSON [extra flags]" >&2
  echo "Or: $0 --image IMAGE --checkpoint CKPT --output OUTPUT --single-crop" >&2
  exit 2
fi
image="$1"; checkpoint="$2"; output="$3"; boxes="$4"; shift 4
exec "$PYTHON_CMD" inference_classification.py --image "$image" --checkpoint "$checkpoint" --output "$output" --boxes-json "$boxes" "$@"
