#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON_CMD:-python}" inference_classification.py "$@"
