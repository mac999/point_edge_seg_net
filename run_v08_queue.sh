#!/usr/bin/env bash
# ============================================================================
# v0.8.x follow-up queue — RESTARTABLE. Safe to re-run after a crash/reboot:
#     cd ~/project/point_edge_seg_net && nohup ./run_v08_queue.sh >> v08_queue.log 2>&1 &
# State lives in logs/v08_queue.state (KEY=VALUE per line). Each step:
#   - already scored           -> skip
#   - run dir + final_model    -> skip training, score
#   - run dir + checkpoint     -> RESUME training from checkpoint (same schedule)
#   - nothing                  -> fresh launch
#
# Steps (sequential, single GPU job at a time, 300 W cap respected):
#   Q1  E5: E4 recipe (stencil r2 + diff) x 600 epochs      (~5-6 h)
#   Q2  TTA scoring (5 scales x flip = 10 views) of E5 final and E4 final
#   Q3  E6: room-mode training (whole voxelized rooms) with v2.1, 300 epochs
#   Q4  E7: 2 cm voxel cache + training (fine geometry), 150 epochs
# ============================================================================
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-/home/tom3/miniconda3/envs/pesn/bin/python}"
STATE=logs/v08_queue.state
touch "$STATE"
get() { grep -E "^$1=" "$STATE" | tail -1 | cut -d= -f2-; }
put() { echo "$1=$2" >> "$STATE"; echo "=== [STATE] $1=$2 $(date '+%F %T') ==="; }

newest_run() { ls -td logs/*/ 2>/dev/null | head -1 | sed 's:/$::'; }

train() { # $1 state_key  $2... train_model args
    local key=$1; shift
    local run=$(get "$key")
    if [ -n "$run" ] && [ -f "$run/final_model.pth" ]; then
        echo "=== [$key] already trained -> $run ==="; return 0
    fi
    if [ -n "$run" ] && [ -f "$run/checkpoint.pth" ]; then
        echo "=== [$key] RESUMING $run $(date '+%F %T') ==="
        local before=$(newest_run)
        "$PYTHON" train_model.py "$@" --resume "$run/checkpoint.pth" || return 1
        local newrun=$(newest_run)   # --resume writes into a NEW logs/<ts>/ dir
        [ "$newrun" != "$before" ] && put "$key" "$newrun"
        return 0
    fi
    echo "=== [$key] LAUNCH $(date '+%F %T') ==="
    local before=$(newest_run)
    "$PYTHON" train_model.py "$@" || return 1
    local newrun=$(newest_run)
    [ "$newrun" != "$before" ] && put "$key" "$newrun"
    return 0
}

score() { # $1 tag  $2 rundir  $3 out_suffix  $4... evaluate_full extra args
    local tag=$1 run=$2 suf=$3; shift 3
    [ -n "$run" ] || { echo "=== [$tag] no run dir ==="; return; }
    for W in best_model.pth final_model.pth; do
        [ -f "$run/$W" ] || continue
        local OUT="$run/test_full_${W%.pth}${suf}.json"
        [ -f "$OUT" ] && { echo "=== [$tag] $W$suf already scored ==="; continue; }
        if "$PYTHON" evaluate_full.py --model_weights "$run/$W" \
                --config model_params_room.json "$@" --out "$OUT"; then
            local m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [$tag] RESULT ${W}${suf}: mIoU=${m} (record 62.03) run=$run $(date '+%F %T') ==="
        else
            echo "=== [$tag] EVAL FAILED ${W}${suf} $(date '+%F %T') ==="
        fi
    done
}

COMMON_V2="--arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff \
    --enc_channels 64,192,320,448 --bottleneck_dim 256"
EVAL_CHUNK="--mode chunk --sampler grid --block_size 20480 --core_max 12288 --halo 1.0 \
    --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff \
    --enc_channels 64,192,320,448 --bottleneck_dim 256"

echo "############ V08 QUEUE (re)start $(date '+%F %T') ############"

# ---------- Q1: E5 long run (600 epochs, E4 recipe) ----------
train E5_RUN \
    --config model_params_room.json \
    --processed_data_path ./processed_s3dis --block_data_path ./chunk_s3dis \
    --block_size 20480 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5 \
    --num_epochs 600 $COMMON_V2 \
    --batch_size 4 --val_batch_size 4 --learning_rate 0.003 \
    --block_mode column --sampler grid \
    --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0 \
    || echo "=== [E5] TRAIN FAILED ==="
score E5 "$(get E5_RUN)" "" $EVAL_CHUNK

# ---------- Q2: TTA scoring (10 views) of E5 and E4 finals ----------
score E5-TTA "$(get E5_RUN)" "_tta" $EVAL_CHUNK --tta 5 --tta_flip
score E4-TTA "logs/20260815_153706" "_tta" $EVAL_CHUNK --tta 5 --tta_flip

# ---------- Q3: E6 room-mode training ----------
train E6_RUN \
    --config model_params_room.json \
    --processed_data_path ./processed_s3dis \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5 \
    --num_epochs 300 $COMMON_V2 \
    --block_mode room --room_data_path ./room_s3dis \
    --batch_size 4 --val_batch_size 4 --learning_rate 0.003 --sampler grid \
    --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0 \
    || echo "=== [E6] TRAIN FAILED ==="
score E6 "$(get E6_RUN)" "" --mode room --room_grid 0.04 --sampler grid \
    --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff \
    --enc_channels 64,192,320,448 --bottleneck_dim 256

# ---------- Q4: E7 2cm voxel cache + training ----------
if [ ! -d chunk_s3dis_2cm ] || [ -z "$(ls chunk_s3dis_2cm 2>/dev/null | head -1)" ]; then
    echo "=== [E7] building 2cm chunk cache $(date '+%F %T') ==="
    "$PYTHON" - <<'PYEOF'
from voxel_chunk import prepare_chunk_cache
prepare_chunk_cache('./processed_s3dis', './chunk_s3dis_2cm',
                    ['Area_1','Area_2','Area_3','Area_4','Area_5','Area_6'], 'Area_5',
                    grid=0.02, core_max=24576, halo=1.0, block_size=49152, feature_dim=7)
PYEOF
fi
train E7_RUN \
    --config model_params_room.json \
    --processed_data_path ./processed_s3dis --block_data_path ./chunk_s3dis_2cm \
    --block_size 49152 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5 \
    --num_epochs 150 $COMMON_V2 \
    --v2_base_grid 0.02 --v2_pool_grids 0.04,0.08,0.16 \
    --batch_size 2 --val_batch_size 2 --learning_rate 0.003 \
    --block_mode column --sampler grid \
    --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0 \
    || echo "=== [E7] TRAIN FAILED ==="
score E7 "$(get E7_RUN)" "" --mode chunk --sampler grid --block_size 49152 \
    --core_max 24576 --halo 1.0 --room_grid 0.02 \
    --arch v2 --v2_neighbors stencil --v2_stencil 2 --v2_diff \
    --v2_base_grid 0.02 --v2_pool_grids 0.04,0.08,0.16 \
    --enc_channels 64,192,320,448 --bottleneck_dim 256

echo "############ V08 QUEUE COMPLETE $(date '+%F %T') ############"
