#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$ROOT/src/semantic/train.py" --data "${1:?prepared Assignment II source tiles required}" --out "${2:?new run directory required}" "${@:3}"
