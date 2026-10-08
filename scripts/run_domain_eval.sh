#!/usr/bin/env bash
# ============================================================
#  Score a checkpoint on the geometry its domain trained it on.
#
#  --domain reads the block geometry, voxel lattice and
#  architecture back out of domains/<name>.json. This is not a
#  convenience: w6 and w12 were once scored at the default 2 m
#  window and full resolution and came out 3.9 mIoU low, which
#  produced three wrong conclusions before the mismatch was found.
#
#  Usage:
#    ./run_domain_eval.sh <domain> <checkpoint.pth> [protocol] [out.json]
#
#  protocol (default overlap_mirror, the reported one):
#    single          one view, stride = window     comparable to the paper baselines
#    overlap         stride = window/2
#    mirror          2 views (identity + mirrored)
#    overlap_mirror  both                          +0.8 mIoU over single on w6
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root: the entry points live there

DOMAIN="${1:-}"
CKPT="${2:-}"
PROTOCOL="${3:-overlap_mirror}"
if [ -z "$DOMAIN" ] || [ -z "$CKPT" ]; then
    echo "usage: $0 <domain> <checkpoint.pth> [protocol] [out.json]" >&2
    exit 1
fi
[ -f "$CKPT" ] || { echo "[ERROR] no such checkpoint: $CKPT" >&2; exit 1; }
OUT="${4:-$(dirname "$CKPT")/score_${PROTOCOL}_$(basename "${CKPT%.pth}").json}"

PYTHON="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== scoring $CKPT as domain '$DOMAIN', protocol '$PROTOCOL' ==="
exec "$PYTHON" evaluate_full.py \
    --domain "$DOMAIN" --protocol "$PROTOCOL" \
    --model_weights "$CKPT" --out "$OUT" "${@:5}"
