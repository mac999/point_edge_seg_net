#!/usr/bin/env bash
# ============================================================
#  The headline SemanticBridge run: the 6 m / 4 cm recipe trained
#  on every labelled block of the 15 training bridges.
#
#  Why this is a separate recipe from bridge_w6
#  -------------------------------------------
#  bridge_w6 holds out 20% of the blocks for validation and early
#  stopping. The published benchmark baselines were trained without
#  a validation set, so that held-out fifth is a protocol difference
#  and not a property of the model. bridge_w6_all removes it:
#    --val_in_train         every labelled block goes into training
#    num_epochs 400         fixed in advance, so nothing is selected
#    early_stop_patience    effectively disabled
#  The per-epoch validation numbers printed during training are
#  therefore IN-SAMPLE. They must not be used to pick a checkpoint.
#  Report final_model.pth. best_model.pth is not released for this
#  run for exactly that reason.
#
#  Measured (84,153,822 points, official 15/5 split)
#  -------------------------------------------------
#     single          mIoU 70.04   OA 91.22   <- protocol matches the baselines
#     overlap_mirror  mIoU 70.74   OA 91.82   <- headline; votes, baselines do not
#  Repeating a pass moves the last digit: the single-view figure has been
#  measured at both 70.04 and 70.05 on this checkpoint. Spread of about
#  0.01 mIoU is normal here, so treat differences under 0.05 as noise.
#  For reference, the same recipe with the validation split held out
#  (bridge_w6) scores 69.81 single-view, and the two-scale fusion that
#  gained 0.68 there does not help here: fusing this checkpoint with the
#  12 m one gives 70.50, below voting alone.
#
#  Usage
#  -----
#    ./run_bridge_w6_all.sh              score the released checkpoint
#    ./run_bridge_w6_all.sh --train      train from scratch first (~54 h on one card)
#    ./run_bridge_w6_all.sh --fusion     also score the 12 m partner and fuse
#
#  Environment: PYTHON, CUDA_VISIBLE_DEVICES, OUTDIR.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root: the entry points live there

DO_TRAIN=0
DO_FUSION=0
for arg in "$@"; do
    case "$arg" in
        --train)  DO_TRAIN=1 ;;
        --fusion) DO_FUSION=1 ;;
        # print the header block, however long it grows
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

OUTDIR="${OUTDIR:-reproduce_w6_all}"
CKPT="logs/20261018_163423_bridge_w6_all/final_model.pth"
CKPT12="logs/20261001_050040_bridge_w12/final_model.pth"
mkdir -p "$OUTDIR"

if [ "$DO_TRAIN" = 1 ]; then
    echo "=== training bridge_w6_all (400 epochs, no early stop) on GPU ${CUDA_VISIBLE_DEVICES} ==="
    echo "    the validation numbers printed below are in-sample; do not select on them"
    "$PYTHON" train_model.py --domain bridge_w6_all
    # train_model.py writes logs/<timestamp>_bridge_w6_all/; pick the newest one up.
    CKPT="$(ls -dt logs/*_bridge_w6_all | head -1)/final_model.pth"
    echo "=== trained checkpoint: $CKPT ==="
fi

[ -f "$CKPT" ] || { echo "[ERROR] no such checkpoint: $CKPT" >&2; exit 1; }

score () {  # <tag> <protocol> [extra flags...]
    local tag="$1" protocol="$2"; shift 2
    echo
    echo "=== $tag (protocol $protocol) ==="
    # --domain bridge_w6: the scoring geometry is the recipe's, not the training variant's.
    "$PYTHON" evaluate_full.py --domain bridge_w6 --protocol "$protocol" \
        --model_weights "$CKPT" --out "$OUTDIR/$tag.json" --overwrite "$@"
}

score single          single
score overlap_mirror  overlap_mirror --save_probs "$OUTDIR/probs_w6all_om"

if [ "$DO_FUSION" = 1 ]; then
    if [ -f "$CKPT12" ]; then
        echo
        echo "=== 12 m partner, same protocol, then equal-weight point-wise fusion ==="
        "$PYTHON" evaluate_full.py --domain bridge_w12 --protocol overlap_mirror \
            --model_weights "$CKPT12" --out "$OUTDIR/w12_overlap_mirror.json" --overwrite \
            --save_probs "$OUTDIR/probs_w12_om"
        "$PYTHON" tools/fuse_votes.py --probs "$OUTDIR/probs_w6all_om" "$OUTDIR/probs_w12_om" \
            --processed_data_path bridge/processed --test_area test \
            --config configs/model_params_semanticbridge.json \
            --out "$OUTDIR/fusion_w6all_w12.json" --overwrite
    else
        echo "[skip] no 12 m checkpoint at $CKPT12; fusion not run"
    fi
fi

echo
echo "=== summary ==="
"$PYTHON" - "$OUTDIR" <<'PY'
import json, os, sys
d = sys.argv[1]
for tag in ('single', 'overlap_mirror', 'w12_overlap_mirror', 'fusion_w6all_w12'):
    p = os.path.join(d, tag + '.json')
    if not os.path.exists(p):
        continue
    m = json.load(open(p))['overall_metrics']
    print(f"{tag:20s} mIoU {m['mIoU']*100:6.2f}  OA {m['accuracy']*100:6.2f}  "
          f"({m['total_points']:,} points)")
PY
