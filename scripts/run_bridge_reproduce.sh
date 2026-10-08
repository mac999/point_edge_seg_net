#!/usr/bin/env bash
# ============================================================
#  Reproduce every reported SemanticBridge figure from the
#  released w6 checkpoint. Scoring only -- no training.
#
#  Expected (84,153,822 points, official 15/5 split):
#     single          mIoU 69.81   OA 91.61    <- compare to the paper baselines
#     overlap_mirror  mIoU 70.64   OA 91.94    <- reported headline
#  Cross-sensor, the three bridges scanned with both (13/17/19):
#     TLS             mIoU 76.16   OA 92.82
#     MLS             mIoU 65.69   OA 90.65    (-10.47 domain gap)
#
#  Usage:  ./run_bridge_reproduce.sh [checkpoint.pth]
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root: the entry points live there

CKPT="${1:-logs/20260929_150153_bridge_w6/final_model.pth}"
[ -f "$CKPT" ] || { echo "[ERROR] no such checkpoint: $CKPT" >&2; exit 1; }
OUTDIR="${OUTDIR:-reproduce}"
mkdir -p "$OUTDIR"

PYTHON="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

score () {  # <tag> <protocol> [extra flags...]
    local tag="$1" protocol="$2"; shift 2
    echo
    echo "=== $tag (protocol $protocol) ==="
    "$PYTHON" evaluate_full.py --domain bridge_w6 --protocol "$protocol" \
        --model_weights "$CKPT" --out "$OUTDIR/$tag.json" --overwrite "$@"
}

score tls5_single          single
score tls5_overlap_mirror  overlap_mirror
# Cross-sensor pair: the same three bridges, scanned TLS (rtc) and MLS (blk). Both are
# scored with the reported protocol, so the gap is a sensor effect and nothing else.
score tls3_overlap_mirror  overlap_mirror --processed_data_path bridge/processed_tls3
score mls3_overlap_mirror  overlap_mirror --processed_data_path bridge/processed_mls

echo
echo "=== summary ==="
"$PYTHON" - "$OUTDIR" <<'PY'
import json, os, sys
d = sys.argv[1]
for tag in ('tls5_single', 'tls5_overlap_mirror', 'tls3_overlap_mirror', 'mls3_overlap_mirror'):
    p = os.path.join(d, tag + '.json')
    if not os.path.exists(p):
        continue
    m = json.load(open(p))['overall_metrics']
    print(f"{tag:22s} mIoU {m['mIoU']*100:6.2f}  OA {m['accuracy']*100:6.2f}  "
          f"({m['total_points']:,} points)")
PY
