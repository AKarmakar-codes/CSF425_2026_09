#!/usr/bin/env bash
set -euo pipefail
# Artifact name: run_instance.sh (Assignment II Section 7)
ROOT="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/best_instance.ckpt}"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$ROOT/inference_instance.py" --checkpoint "$CHECKPOINT" --images "${1:?image or directory required}" --out "${2:?output directory required}" "${@:3}"
