#!/usr/bin/env bash
# ============================================================
#  The released SemanticBridge run: the 12 m / 4 cm recipe trained
#  on every labelled block of the 15 training bridges.
#
#  Why 12 m / 4 cm
#  ---------------
#  The context sweep scales the voxel with the window (6 m / 4 cm,
#  12 m / 8 cm, ...) and peaks at 6 m. Two controls that change one
#  of them at a time (bridge_w6_v8, bridge_w12_v4) show that peak is
#  the coarser voxel's: at a fixed 4 cm voxel the 12 m window scores
#  70.82 single-view against 69.81, mostly on abutments.
#
#  Like bridge_w6_all, this recipe folds the held-out validation
#  blocks back into training (--val_in_train), fixes 400 epochs in
#  advance and disables early stopping. The per-epoch validation
#  numbers printed during training are therefore IN-SAMPLE and must
#  not be used to pick a checkpoint. Report final_model.pth.
#
#  Measured (84,153,822 points, official 15/5 split)
#  -------------------------------------------------
#     single          mIoU 70.80   OA 92.44   <- protocol matches the baselines
#     overlap_mirror  mIoU 71.87   OA 92.75   <- votes; baselines do not
#  Treat differences under 0.05 mIoU as scoring noise. The seed spread,
#  recorded in logs/W12_V4_ALL.md, is larger than that.
#
#  Usage
#  -----
#    ./run_bridge_w12_v4_all.sh            score the released checkpoint
#    ./run_bridge_w12_v4_all.sh --train    train from scratch first (~4 days on one card)
#
#  Training peaks at about 44 GB of GPU memory.
#  Environment: PYTHON, CUDA_VISIBLE_DEVICES, OUTDIR.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root: the entry points live there

DO_TRAIN=0
for arg in "$@"; do
    case "$arg" in
        --train)  DO_TRAIN=1 ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown option: $arg" >&2; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch. Set PYTHON=/path/to/python." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTDIR="${OUTDIR:-reproduce_w12_v4_all}"
CKPT="logs/20261111_215452_bridge_w12_v4_all/final_model.pth"
mkdir -p "$OUTDIR"

if [ "$DO_TRAIN" = 1 ]; then
    echo "=== training bridge_w12_v4_all (400 epochs, no early stop) on GPU ${CUDA_VISIBLE_DEVICES} ==="
    echo "    the validation numbers printed below are in-sample; do not select on them"
    "$PYTHON" train_model.py --domain bridge_w12_v4_all
    # train_model.py writes logs/<timestamp>_bridge_w12_v4_all/; pick the newest one up.
    CKPT="$(ls -dt logs/*_bridge_w12_v4_all | head -1)/final_model.pth"
    echo "=== trained checkpoint: $CKPT ==="
fi

[ -f "$CKPT" ] || { echo "[ERROR] no such checkpoint: $CKPT" >&2; exit 1; }

score () {  # <tag> <protocol> [extra flags...]
    local tag="$1" protocol="$2"; shift 2
    echo
    echo "=== $tag (protocol $protocol) ==="
    # --domain bridge_w12_v4: the scoring geometry is the recipe's, not the training variant's.
    "$PYTHON" evaluate_full.py --domain bridge_w12_v4 --protocol "$protocol" \
        --model_weights "$CKPT" --out "$OUTDIR/$tag.json" --overwrite "$@"
}

score single          single
score overlap_mirror  overlap_mirror
# Cross-sensor pair: the same three bridges, scanned TLS and MLS, single view.
score tls3_single     single --processed_data_path bridge/processed_tls3
score mls3_single     single --processed_data_path bridge/processed_mls

echo
echo "=== summary ==="
"$PYTHON" - "$OUTDIR" <<'PY'
import json, os, sys
d = sys.argv[1]
for tag in ('single', 'overlap_mirror', 'tls3_single', 'mls3_single'):
    p = os.path.join(d, tag + '.json')
    if not os.path.exists(p):
        continue
    m = json.load(open(p))['overall_metrics']
    print(f"{tag:20s} mIoU {m['mIoU']*100:6.2f}  OA {m['accuracy']*100:6.2f}  "
          f"({m['total_points']:,} points)")
PY
