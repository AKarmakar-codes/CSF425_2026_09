#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA="${1:?prepared Assignment II source tiles required}"
OUT="${2:?ablation output directory required}"
EPOCHS="${3:-60}"
DEVICE="${4:-cuda:0}"
PRETRAIN_FLAG=()
if [[ "${5:-false}" == "true" ]]; then PRETRAIN_FLAG=(--imagenet-pretrained); fi

echo "=== Candidate 1: Full Masked-Attention (Sweet Spot 18.5M, h=256, l=6, q=100) ==="
"$PYTHON" "$ROOT/src/semantic/train.py" --data "$DATA" --out "$OUT/full_masked_q100_h256_l6" \
  --seed 42 --queries 100 --hidden-dim 256 --layers 6 --memory-size 32 --boundary-weight 1.0 \
  --epochs "$EPOCHS" --device "$DEVICE" "${PRETRAIN_FLAG[@]}"

echo "=== Candidate 2: Ablate Masked-Attention (Unmasked Cross-Attention, h=256, l=6, q=100) ==="
"$PYTHON" "$ROOT/src/semantic/train.py" --data "$DATA" --out "$OUT/ablate_unmasked_q100_h256_l6" \
  --seed 42 --queries 100 --hidden-dim 256 --layers 6 --memory-size 32 --boundary-weight 1.0 \
  --disable-masked-attention --epochs "$EPOCHS" --device "$DEVICE" "${PRETRAIN_FLAG[@]}"

echo "=== Candidate 3: Compact Variant (h=128, l=6, q=100, 13.1M params) ==="
"$PYTHON" "$ROOT/src/semantic/train.py" --data "$DATA" --out "$OUT/compact_q100_h128_l6" \
  --seed 42 --queries 100 --hidden-dim 128 --layers 6 --memory-size 32 --boundary-weight 1.0 \
  --epochs "$EPOCHS" --device "$DEVICE" "${PRETRAIN_FLAG[@]}"

echo "=== Selecting Candidate based strictly on Source Validation ==="
"$PYTHON" "$ROOT/src/semantic/select_candidate.py" \
  --runs "$OUT/full_masked_q100_h256_l6" "$OUT/ablate_unmasked_q100_h256_l6" "$OUT/compact_q100_h128_l6" \
  --output "$OUT/selection.json" --promote-dir "$ROOT/checkpoints"
