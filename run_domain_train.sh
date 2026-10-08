#!/usr/bin/env bash
# ============================================================
#  Train one domain recipe end to end.
#
#  Every per-model setting -- block geometry, voxel lattice,
#  architecture flags, schedule, loss options -- lives in
#  domains/<name>.json, so this script carries none of them and
#  two models never drift apart because a flag was retyped.
#
#  Usage:   ./run_domain_train.sh <domain> [extra train_model.py flags...]
#
#  Bridge domains (SemanticBridge, 15/5 official split):
#    bridge          2 m window, full resolution      baseline, mIoU 65.23
#    bridge_w6       6 m window, 4 cm voxels          BEST, mIoU 70.64
#    bridge_w12      12 m window, 8 cm voxels         69.03
#    bridge_w24      24 m window, 16 cm voxels        67.11 (coarse)
#    bridge_w6_gpos  w6 + global-position channels    68.41  REJECTED -2.06
#    bridge_w6_sol   w6 + structure-oriented loss     70.14  REJECTED -0.50
#  Indoor:
#    room            S3DIS chunk recipe
#
#  Logs go to a fresh <timestamp>_<domain> directory; an existing
#  run is never overwritten.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

DOMAIN="${1:-}"
if [ -z "$DOMAIN" ]; then
    echo "usage: $0 <domain> [extra flags...]" >&2
    echo "available:" >&2
    ls domains/*.json | sed 's|domains/||; s|\.json$||; s|^|  |' >&2
    exit 1
fi
shift

PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch. Set PYTHON=/path/to/python." >&2
    exit 1
fi

# expandable_segments keeps the allocator from fragmenting on the long runs; one GPU at a
# time is deliberate (these recipes are sized for a single card).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== training domain '$DOMAIN' on GPU ${CUDA_VISIBLE_DEVICES} ==="
exec "$PYTHON" train_model.py --domain "$DOMAIN" "$@"
