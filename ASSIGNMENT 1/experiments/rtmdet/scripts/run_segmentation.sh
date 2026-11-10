#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"${PCB_PYTHON:-python}" "$HERE/inference_segmentation.py" "$@"
