#!/usr/bin/env bash
# ============================================================
#  Capacity-rebalance experiment, queued behind the running one.
#  Picks its feature set from whichever of the two chunk recipes actually won:
#    - 10D invariant geometry (chunk_s3dis_inv) if the running candidate beats 58.89
#    - 7D                     (chunk_s3dis)     otherwise
#  so the architecture change is measured on top of the best data recipe, not a guess.
# ============================================================
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
HERE=$(pwd -P)
BEST_MIOU=58.89          # voxel-chunk @132 epochs, logs/20260730_121057

newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }
gpu_job_running() {
    local pid exe cwd
    for pid in $(pgrep -f 'train_model\.py|evaluate_full\.py' 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || continue
        case "$exe" in *python*) ;; *) continue ;; esac
        cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null) || continue
        [ "$cwd" = "$HERE" ] && return 0
    done
    return 1
}

while gpu_job_running; do sleep 60; done

# Which feature set won?
INV=$(grep -oE "mIoU=[0-9.]+" chunkinv.log 2>/dev/null | tail -1 | cut -d= -f2)
if [ -n "$INV" ] && awk "BEGIN{exit !($INV > $BEST_MIOU)}"; then
    CACHE=./chunk_s3dis_inv ; CONF=model_params.json      ; TAG="10D-invariant (${INV} > ${BEST_MIOU})"
else
    CACHE=./chunk_s3dis     ; CONF=model_params_room.json ; TAG="7D (invariant scored ${INV:-n/a}, not better than ${BEST_MIOU})"
fi
echo "=== REBAL START $(date '+%F %T') | base features: $TAG ==="

before=$(newest_run)
"$PYTHON" train_model.py \
    --config "$CONF" \
    --processed_data_path ./processed_s3dis \
    --block_data_path "$CACHE" \
    --block_size 20480 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 \
    --test_area Area_5 \
    --num_epochs 150 \
    --enc_channels 64,192,320,448 \
    --bottleneck_dim 256 \
    --batch_size 4 \
    --val_batch_size 4 \
    --learning_rate 0.003 \
    --block_mode column \
    --sampler grid \
    --focal_gamma 2.0 \
    --oversample_rare 1.0 \
    --aug_preset strong \
    --cooldown_sec 0
status=$?

run=$(newest_run)
if [ $status -eq 0 ] && [ -n "$run" ] && [ "$run" != "$before" ] && [ -f "${run}best_model.pth" ]; then
    echo "=== [REBAL] TRAIN DONE -> $run $(date '+%F %T') ==="
    if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth" \
            --config "$CONF" --mode chunk --sampler grid \
            --block_size 20480 --core_max 12288 --halo 1.0 \
            --enc_channels 64,192,320,448 --bottleneck_dim 256; then
        miou=$("$PYTHON" -c "import json;print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
        echo "=== [REBAL] RESULT full-protocol mIoU=${miou} (best so far ${BEST_MIOU}) run=${run} $(date '+%F %T') ==="
    else
        echo "=== [REBAL] EVAL FAILED $(date '+%F %T') ==="
    fi
else
    echo "=== [REBAL] TRAIN FAILED (exit $status) $(date '+%F %T') ==="
fi
echo "=== REBAL COMPLETE $(date '+%F %T') ==="
