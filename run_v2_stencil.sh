#!/usr/bin/env bash
# v2.1 experiments: exact grid-stencil neighbours + feature-difference term.
#   E3: --v2_neighbors stencil --v2_stencil 1  (K=27,  ~9 surface nbrs, 45 ms/step)
#   E4: --v2_neighbors stencil --v2_stencil 2  (K=125, ~25 surface nbrs ~= v1 kNN-32, 170 ms/step)
# Hypothesis from E1/E2: the serialized windows' APPROXIMATE neighbours are the main
# quality loss; exact metric neighbours should recover door/board/chair.
# PID capture: launch() writes $! to a pidfile from ITS OWN shell — the $(...) form
# ran the background job in a subshell and broke `wait` (exit 127) last time.
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-/home/tom3/miniconda3/envs/pesn/bin/python}"

launch() { # $1 stencil_radius  $2 logfile  $3 pidfile
    "$PYTHON" train_model.py \
        --config model_params_room.json \
        --processed_data_path ./processed_s3dis \
        --block_data_path ./chunk_s3dis \
        --block_size 20480 \
        --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5 \
        --num_epochs 150 \
        --arch v2 --v2_neighbors stencil --v2_stencil "$1" --v2_diff \
        --enc_channels 64,192,320,448 --bottleneck_dim 256 \
        --batch_size 4 --val_batch_size 4 --learning_rate 0.003 \
        --block_mode column --sampler grid \
        --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0 \
        > "$2" 2>&1 &
    echo $! > "$3"
}

logdir_of() {
    local d=''
    for _ in $(seq 1 60); do
        d=$(grep -aoE "Logging to directory: logs/[0-9_]+" "$1" 2>/dev/null | head -1 | awk '{print $4}')
        [ -n "$d" ] && break
        sleep 5
    done
    echo "$d"
}

echo "=== STENCIL START $(date '+%F %T') ==="
launch 1 v2_e3.log /tmp/claude-1000/e3.pid; P3=$(cat /tmp/claude-1000/e3.pid); RUN3=$(logdir_of v2_e3.log)
launch 2 v2_e4.log /tmp/claude-1000/e4.pid; P4=$(cat /tmp/claude-1000/e4.pid); RUN4=$(logdir_of v2_e4.log)
echo "=== E3 (r1) pid=$P3 -> $RUN3 | E4 (r2) pid=$P4 -> $RUN4 ==="
wait "$P3"; S3=$?
wait "$P4"; S4=$?
echo "=== E3 exit=$S3  E4 exit=$S4  $(date '+%F %T') ==="

score() { # $1 tag  $2 rundir  $3 radius
    [ -n "$2" ] || { echo "=== [$1] NO RUN DIR ==="; return; }
    for W in best_model.pth final_model.pth; do
        [ -f "$2/${W}" ] || { echo "=== [$1] missing ${W} ==="; continue; }
        OUT="$2/test_full_${W%.pth}.json"
        if "$PYTHON" evaluate_full.py --model_weights "$2/${W}" \
                --config model_params_room.json --mode chunk --sampler grid \
                --block_size 20480 --core_max 12288 --halo 1.0 \
                --arch v2 --v2_neighbors stencil --v2_stencil "$3" --v2_diff \
                --enc_channels 64,192,320,448 --bottleneck_dim 256 \
                --out "$OUT"; then
            m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [$1] RESULT ${W}: mIoU=${m} (v1 58.82 | serial-best 49.35) run=$2 ==="
        else
            echo "=== [$1] EVAL FAILED ${W} ==="
        fi
    done
}
score E3 "$RUN3" 1
score E4 "$RUN4" 2
echo "=== STENCIL COMPLETE $(date '+%F %T') ==="
