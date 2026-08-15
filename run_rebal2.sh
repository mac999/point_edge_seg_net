#!/usr/bin/env bash
# Capacity rebalance on the refactored, train/eval-consistent pipeline.
#   (64,192,320,448) encoder + 256-wide bottleneck, 7D features, 150 epochs.
# Baseline to beat: 58.92 (same recipe at 5.73M params, re-scored consistently).
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }
echo "=== REBAL2 START $(date '+%F %T') ==="
before=$(newest_run)
"$PYTHON" train_model.py \
    --config model_params_room.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./chunk_s3dis \
    --block_size 20480 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5 \
    --num_epochs 150 \
    --enc_channels 64,192,320,448 --bottleneck_dim 256 \
    --batch_size 4 --val_batch_size 4 --learning_rate 0.003 \
    --block_mode column --sampler grid \
    --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0
st=$?
run=$(newest_run)
if [ $st -eq 0 ] && [ -n "$run" ] && [ "$run" != "$before" ] && [ -f "${run}best_model.pth" ]; then
    echo "=== [REBAL2] TRAIN DONE -> $run $(date '+%F %T') ==="
    if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth" \
            --config model_params_room.json --mode chunk --sampler grid \
            --block_size 20480 --core_max 12288 --halo 1.0 \
            --enc_channels 64,192,320,448 --bottleneck_dim 256; then
        m=$("$PYTHON" -c "import json;print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
        echo "=== [REBAL2] RESULT full-protocol mIoU=${m} (baseline 58.92) run=${run} $(date '+%F %T') ==="
    else
        echo "=== [REBAL2] EVAL FAILED $(date '+%F %T') ==="
    fi
else
    echo "=== [REBAL2] TRAIN FAILED (exit $st) $(date '+%F %T') ==="
fi
echo "=== REBAL2 COMPLETE $(date '+%F %T') ==="
