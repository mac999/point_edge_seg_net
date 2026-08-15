#!/usr/bin/env bash
# Resume the interrupted REBAL2 run (logs/20260813_082752) at epoch 140/150.
#
# The 2026-08-14 08:08 upgrade of the shared `dl` env to torch 2.13.0+cu130 removed the
# PyG stack, so this run is pinned to its own env instead: conda `pesn` carries
# torch 2.7.1+cu128 + torch_cluster/scatter/sparse pt27cu128 wheels, matching the stack
# the checkpoint was produced on. `dl` is left untouched for the LLM tooling installed there.
#
# checkpoint.pth carries model, optimizer moments, cosine position, AMP scaler and full
# history, so --resume continues the SAME schedule in the SAME log dir with nothing lost.
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-/home/tom3/miniconda3/envs/pesn/bin/python}"
FROM=logs/20260813_082752/    # checkpoint we resume FROM (epoch 139)
BASELINE=58.92                # full-protocol mIoU to beat (logs/20260730_121057)

# train_model.py stamps a FRESH logs/<timestamp>/ on every start, --resume included
# (log_dir is built from datetime at train_model.py:341 before the checkpoint is read).
# So epochs 140-150 and the new best_model.pth land in a NEW dir, not in $FROM -- the
# "results stay in $RUN" comment in run_resume_queue.sh is wrong. Resolve the real dir
# after training and score THAT, or we would re-score the stale epoch-134 weights.
newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }
before=$(newest_run)

echo "=== REBAL2-RESUME START $(date '+%F %T') ==="
echo "=== python: $PYTHON ==="
"$PYTHON" train_model.py \
    --config model_params_room.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./chunk_s3dis \
    --block_size 20480 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5 \
    --num_epochs 150 \
    --resume "${FROM}checkpoint.pth" \
    --enc_channels 64,192,320,448 --bottleneck_dim 256 \
    --batch_size 4 --val_batch_size 4 --learning_rate 0.003 \
    --block_mode column --sampler grid \
    --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0
st=$?

RUN=$(newest_run)
if [ $st -eq 0 ] && [ -n "$RUN" ] && [ "$RUN" != "$before" ] && [ -f "${RUN}best_model.pth" ]; then
    echo "=== [REBAL2-RESUME] TRAIN DONE -> $RUN $(date '+%F %T') ==="
    if "$PYTHON" evaluate_full.py --model_weights "${RUN}best_model.pth" \
            --config model_params_room.json --mode chunk --sampler grid \
            --block_size 20480 --core_max 12288 --halo 1.0 \
            --enc_channels 64,192,320,448 --bottleneck_dim 256; then
        m=$("$PYTHON" -c "import json;print(f\"{json.load(open('${RUN}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
        echo "=== [REBAL2-RESUME] RESULT full-protocol mIoU=${m} (baseline ${BASELINE}) run=${RUN} $(date '+%F %T') ==="
    else
        echo "=== [REBAL2-RESUME] EVAL FAILED $(date '+%F %T') ==="
    fi
else
    echo "=== [REBAL2-RESUME] TRAIN FAILED (exit $st) $(date '+%F %T') ==="
fi
echo "=== REBAL2-RESUME COMPLETE $(date '+%F %T') ==="
