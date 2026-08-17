#!/usr/bin/env bash
# E8: cylindrical anisotropic stencil (directional gating, stages 1+) + beam boost.
#   - --v2_directional: per-(z-level x ring) channel scale/bias on neighbour features,
#     zero-init (starts exactly as E5), yaw-augmentation-safe. Targets window/column/
#     board/beam (vertical structure). Cost: 243 ms/step vs E5 167 ms, peak 8.8 GB.
#   - model_params_room_beam.json: beam class_weight 1.04 -> 3.0 (free rider; Area_5
#     beam GT is only 0.029% so expectations are low — see VERSIONS.md).
#   - 600 epochs, same recipe as E5 otherwise. Baseline to beat: 64.23 (E5 final).
# RESTARTABLE: rerun this script after a crash — resumes from checkpoint via state file.
#     cd ~/project/point_edge_seg_net && nohup ./run_e8.sh >> e8.log 2>&1 &
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-/home/tom3/miniconda3/envs/pesn/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # tame the sawtooth (op note 08-16)
STATE=logs/e8.state
touch "$STATE"
RUN=$(grep -E "^E8_RUN=" "$STATE" | tail -1 | cut -d= -f2-)

ARGS=(--config model_params_room_beam.json
      --processed_data_path ./processed_s3dis --block_data_path ./chunk_s3dis
      --block_size 20480
      --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5
      --num_epochs 600
      --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff --v2_directional
      --enc_channels 64,192,320,448 --bottleneck_dim 256
      --batch_size 4 --val_batch_size 4 --learning_rate 0.003
      --block_mode column --sampler grid
      --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0)

newest_run() { ls -td logs/*/ 2>/dev/null | head -1 | sed 's:/$::'; }

if [ -n "$RUN" ] && [ -f "$RUN/final_model.pth" ]; then
    echo "=== [E8] already trained -> $RUN ==="
elif [ -n "$RUN" ] && [ -f "$RUN/checkpoint.pth" ]; then
    echo "=== [E8] RESUMING $RUN $(date '+%F %T') ==="
    before=$(newest_run)
    "$PYTHON" train_model.py "${ARGS[@]}" --resume "$RUN/checkpoint.pth"
    new=$(newest_run); [ "$new" != "$before" ] && { echo "E8_RUN=$new" >> "$STATE"; RUN=$new; }
else
    echo "=== [E8] LAUNCH $(date '+%F %T') ==="
    before=$(newest_run)
    "$PYTHON" train_model.py "${ARGS[@]}" &
    TPID=$!
    # record the run dir as soon as it exists (crash-resume works from epoch 1)
    for _ in $(seq 1 60); do
        sleep 5; new=$(newest_run)
        [ "$new" != "$before" ] && { echo "E8_RUN=$new" >> "$STATE"; RUN=$new; break; }
    done
    wait $TPID || echo "=== [E8] TRAIN exited nonzero ==="
fi

for W in best_model.pth final_model.pth; do
    [ -f "$RUN/$W" ] || { echo "=== [E8] missing $W ==="; continue; }
    OUT="$RUN/test_full_${W%.pth}.json"
    [ -f "$OUT" ] && continue
    "$PYTHON" evaluate_full.py --model_weights "$RUN/$W" \
        --config model_params_room_beam.json --mode chunk --sampler grid \
        --block_size 20480 --core_max 12288 --halo 1.0 \
        --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff --v2_directional \
        --enc_channels 64,192,320,448 --bottleneck_dim 256 \
        --out "$OUT" \
      && m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")") \
      && echo "=== [E8] RESULT ${W}: mIoU=${m} (E5 baseline 64.23 | record 64.58) $(date '+%F %T') ==="
done
# D4 TTA on final (the proven free +0.35)
OUT="$RUN/test_full_final_d4tta.json"
if [ -f "$RUN/final_model.pth" ] && [ ! -f "$OUT" ]; then
    "$PYTHON" evaluate_full.py --model_weights "$RUN/final_model.pth" \
        --config model_params_room_beam.json --mode chunk --sampler grid \
        --block_size 20480 --core_max 12288 --halo 1.0 \
        --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff --v2_directional \
        --enc_channels 64,192,320,448 --bottleneck_dim 256 \
        --tta_d4 8 --out "$OUT" \
      && m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")") \
      && echo "=== [E8] RESULT final+D4TTA: mIoU=${m} (record 64.58) $(date '+%F %T') ==="
fi
echo "=== E8 COMPLETE $(date '+%F %T') ==="
