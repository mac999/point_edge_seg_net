#!/usr/bin/env bash
# ============================================================
#  Resume the interrupted invariant-geometry run, then score it, then run the
#  capacity-rebalance experiment on whichever feature set actually won.
#
#  The run was stopped at epoch 132/150 to free the GPU. checkpoint.pth carries the
#  model, optimizer moments, cosine position, AMP scaler and full history, so --resume
#  continues the SAME schedule at epoch 133 with nothing lost.
# ============================================================
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
HERE=$(pwd -P)
RUN=logs/20260802_142854/
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
miou_of() { "$PYTHON" -c "import json;print(f\"{json.load(open('$1'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null; }

while gpu_job_running; do sleep 60; done

# ---------- resume the invariant-geometry run ----------
echo "=== RESUME START $(date '+%F %T') ==="
"$PYTHON" train_model.py \
    --config model_params.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./chunk_s3dis_inv \
    --block_size 20480 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 \
    --test_area Area_5 \
    --num_epochs 150 \
    --resume "${RUN}checkpoint.pth" \
    --batch_size 4 --val_batch_size 4 \
    --learning_rate 0.003 \
    --block_mode column --sampler grid \
    --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong \
    --cooldown_sec 0
status=$?

# --resume continues in place, so results stay in $RUN rather than a new log dir.
if [ $status -eq 0 ] && [ -f "${RUN}best_model.pth" ]; then
    echo "=== [CHUNKINV] TRAIN DONE -> $RUN $(date '+%F %T') ==="
    if "$PYTHON" evaluate_full.py --model_weights "${RUN}best_model.pth" \
            --config model_params.json --mode chunk --sampler grid \
            --block_size 20480 --core_max 12288 --halo 1.0; then
        INV=$(miou_of "${RUN}test_full_summary.json")
        echo "=== [CHUNKINV] RESULT full-protocol mIoU=${INV} (best so far ${BEST_MIOU}) run=${RUN} $(date '+%F %T') ==="
    else
        echo "=== [CHUNKINV] EVAL FAILED $(date '+%F %T') ==="
    fi
else
    echo "=== [CHUNKINV] TRAIN FAILED (exit $status) $(date '+%F %T') ==="
fi

# ---------- capacity rebalance, on whichever feature set won ----------
while gpu_job_running; do sleep 60; done
if [ -n "$INV" ] && awk "BEGIN{exit !($INV > $BEST_MIOU)}"; then
    CACHE=./chunk_s3dis_inv ; CONF=model_params.json      ; TAG="10D-invariant (${INV} > ${BEST_MIOU})"
else
    CACHE=./chunk_s3dis     ; CONF=model_params_room.json ; TAG="7D (invariant ${INV:-n/a} did not beat ${BEST_MIOU})"
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
    --enc_channels 64,192,320,448 --bottleneck_dim 256 \
    --batch_size 4 --val_batch_size 4 \
    --learning_rate 0.003 \
    --block_mode column --sampler grid \
    --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong \
    --cooldown_sec 0
rstatus=$?
run=$(newest_run)
if [ $rstatus -eq 0 ] && [ -n "$run" ] && [ "$run" != "$before" ] && [ -f "${run}best_model.pth" ]; then
    echo "=== [REBAL] TRAIN DONE -> $run $(date '+%F %T') ==="
    if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth" \
            --config "$CONF" --mode chunk --sampler grid \
            --block_size 20480 --core_max 12288 --halo 1.0 \
            --enc_channels 64,192,320,448 --bottleneck_dim 256; then
        echo "=== [REBAL] RESULT full-protocol mIoU=$(miou_of ${run}test_full_summary.json) (best so far ${BEST_MIOU}) run=${run} $(date '+%F %T') ==="
    else
        echo "=== [REBAL] EVAL FAILED $(date '+%F %T') ==="
    fi
else
    echo "=== [REBAL] TRAIN FAILED (exit $rstatus) $(date '+%F %T') ==="
fi
echo "=== ALL COMPLETE $(date '+%F %T') ==="
