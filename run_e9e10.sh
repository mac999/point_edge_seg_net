#!/usr/bin/env bash
# E9 + E10 in parallel (restartable). Baseline: E5 final 64.23 / record 64.79.
#   E9:  c1 64->128 — the week-1 "full-resolution starvation" hypothesis, finally
#        tested as a single variable on the E5 recipe. (220 ms/step, 7.8 GB)
#   E10: stencil z-reach 4 (xy 2) — E8 showed per-level gating LOSES column (-4.6);
#        this instead extends the receptive field itself along gravity. (298 ms/step)
# Rerun after crash: cd ~/project/point_edge_seg_net && nohup ./run_e9e10.sh >> e9e10.log 2>&1 &
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-/home/tom3/miniconda3/envs/pesn/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STATE=logs/e9e10.state; touch "$STATE"
get() { grep -E "^$1=" "$STATE" | tail -1 | cut -d= -f2-; }

BASE=(--config model_params_room.json
      --processed_data_path ./processed_s3dis --block_data_path ./chunk_s3dis
      --block_size 20480
      --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5
      --num_epochs 600
      --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff
      --bottleneck_dim 256
      --batch_size 4 --val_batch_size 4 --learning_rate 0.003
      --block_mode column --sampler grid
      --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0)

newest_run() { ls -td logs/*/ 2>/dev/null | head -1 | sed 's:/$::'; }

launch_or_resume() { # $1 key, rest = extra args ; echoes pid, records run dir
    local key=$1; shift
    local run=$(get "$key")
    if [ -n "$run" ] && [ -f "$run/final_model.pth" ]; then
        echo "=== [$key] already trained -> $run ==="; return 1
    fi
    local before=$(newest_run)
    if [ -n "$run" ] && [ -f "$run/checkpoint.pth" ]; then
        echo "=== [$key] RESUMING $run $(date '+%F %T') ==="
        "$PYTHON" train_model.py "${BASE[@]}" "$@" --resume "$run/checkpoint.pth" &
    else
        echo "=== [$key] LAUNCH $(date '+%F %T') ==="
        "$PYTHON" train_model.py "${BASE[@]}" "$@" &
    fi
    local pid=$!
    ( for _ in $(seq 1 120); do sleep 5; n=$(ls -td logs/*/ | head -1 | sed 's:/$::');
        if [ "$n" != "$before" ] && [ -f "$n/training_log.csv" ]; then
            grep -q "^$key=$n\$" "$STATE" || echo "$key=$n" >> "$STATE"; break; fi; done ) &
    echo "$pid"
}

P9=$(launch_or_resume E9_RUN --enc_channels 128,192,320,448 | tail -1)
sleep 30   # let E9 claim its logs/<ts> dir before E10 starts
P10=$(launch_or_resume E10_RUN --enc_channels 64,192,320,448 --v2_stencil_z 4 | tail -1)
echo "=== E9 pid=$P9 | E10 pid=$P10 ==="
[ -n "$P9" ] && [ "$P9" -gt 0 ] 2>/dev/null && wait "$P9"
[ -n "$P10" ] && [ "$P10" -gt 0 ] 2>/dev/null && wait "$P10"
echo "=== trainings done $(date '+%F %T') ==="

score() { # $1 key  $2 extra eval args...
    local key=$1; shift
    local run=$(get "$key")
    [ -n "$run" ] || { echo "=== [$key] no run dir ==="; return; }
    for W in best_model.pth final_model.pth; do
        [ -f "$run/$W" ] || continue
        local OUT="$run/test_full_${W%.pth}.json"
        [ -f "$OUT" ] || "$PYTHON" evaluate_full.py --model_weights "$run/$W" \
            --config model_params_room.json --mode chunk --sampler grid \
            --block_size 20480 --core_max 12288 --halo 1.0 \
            --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff \
            --bottleneck_dim 256 "$@" --out "$OUT" || { echo "=== [$key] EVAL FAILED $W ==="; continue; }
        m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
        echo "=== [$key] RESULT ${W}: mIoU=${m} (E5 64.23 | record 64.79) $(date '+%F %T') ==="
    done
    local OUT="$run/test_full_final_d4tta.json"
    if [ -f "$run/final_model.pth" ] && [ ! -f "$OUT" ]; then
        "$PYTHON" evaluate_full.py --model_weights "$run/final_model.pth" \
            --config model_params_room.json --mode chunk --sampler grid \
            --block_size 20480 --core_max 12288 --halo 1.0 \
            --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff \
            --bottleneck_dim 256 "$@" --tta_d4 8 --out "$OUT" \
          && m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")") \
          && echo "=== [$key] RESULT final+D4TTA: mIoU=${m} (record 64.79) $(date '+%F %T') ==="
    fi
}
score E9_RUN --enc_channels 128,192,320,448
score E10_RUN --enc_channels 64,192,320,448 --v2_stencil_z 4
echo "=== E9E10 COMPLETE $(date '+%F %T') ==="
