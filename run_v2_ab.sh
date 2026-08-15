#!/usr/bin/env bash
# v0.8.0 A/B: serialized meta architecture (model_v2.py) vs v0.6.1 baseline (58.82).
# IDENTICAL recipe to run_rebal2.sh — only --arch v2 differs — so any mIoU delta is
# attributable to the architecture change alone.
# Measured speed: fwd+bwd 30 ms vs 472 ms per batch-4 step; epoch should be I/O-bound.
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-/home/tom3/miniconda3/envs/pesn/bin/python}"
BASELINE=58.82   # v0.6.1 full-protocol mIoU (3.15M params, logs/20260814_233522)

newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }
before=$(newest_run)

echo "=== V2-AB START $(date '+%F %T') ==="
"$PYTHON" train_model.py \
    --config model_params_room.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./chunk_s3dis \
    --block_size 20480 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5 \
    --num_epochs 150 \
    --arch v2 \
    --enc_channels 64,192,320,448 --bottleneck_dim 256 \
    --batch_size 4 --val_batch_size 4 --learning_rate 0.003 \
    --block_mode column --sampler grid \
    --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0
st=$?

RUN=$(newest_run)
if [ $st -eq 0 ] && [ -n "$RUN" ] && [ "$RUN" != "$before" ]; then
    echo "=== [V2-AB] TRAIN DONE -> $RUN $(date '+%F %T') ==="
    # score BOTH best and final: val-metric selection has picked the worse test model
    # before (ep134 57.93 < ep145 58.82), so never trust best_model.pth alone.
    for W in best_model.pth final_model.pth; do
        [ -f "${RUN}${W}" ] || continue
        OUT="${RUN}test_full_${W%.pth}.json"
        if "$PYTHON" evaluate_full.py --model_weights "${RUN}${W}" \
                --config model_params_room.json --mode chunk --sampler grid \
                --block_size 20480 --core_max 12288 --halo 1.0 \
                --arch v2 --enc_channels 64,192,320,448 --bottleneck_dim 256 \
                --out "$OUT"; then
            m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [V2-AB] RESULT ${W}: full-protocol mIoU=${m} (v1 baseline ${BASELINE}) $(date '+%F %T') ==="
        else
            echo "=== [V2-AB] EVAL FAILED for ${W} $(date '+%F %T') ==="
        fi
    done
else
    echo "=== [V2-AB] TRAIN FAILED (exit $st) $(date '+%F %T') ==="
fi
echo "=== V2-AB COMPLETE $(date '+%F %T') ==="
