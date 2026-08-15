#!/usr/bin/env bash
# v2 neighbourhood-widening variants, run IN PARALLEL (each needs ~15 GB; box has 95).
#   E1: --v2_knn 64 --v2_curves 1   (double window on one curve)
#   E2: --v2_knn 64 --v2_curves 2   (two curves x 32 window: independent seams)
# Waits for the running A/B (k32 c1) to finish first, then compares all three against
# the v1 baseline 58.82. Each run's log dir is captured from its own output file, so
# parallel launches cannot be confused by newest_run().
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-/home/tom3/miniconda3/envs/pesn/bin/python}"

echo "=== VARIANTS: waiting for V2-AB to complete $(date '+%F %T') ==="
while ! grep -aq "V2-AB COMPLETE" v2_ab.log 2>/dev/null; do sleep 60; done
echo "=== VARIANTS START $(date '+%F %T') ==="

launch() { # $1 tag  $2 knn  $3 curves  $4 logfile
    "$PYTHON" train_model.py \
        --config model_params_room.json \
        --processed_data_path ./processed_s3dis \
        --block_data_path ./chunk_s3dis \
        --block_size 20480 \
        --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 --test_area Area_5 \
        --num_epochs 150 \
        --arch v2 --v2_knn "$2" --v2_curves "$3" \
        --enc_channels 64,192,320,448 --bottleneck_dim 256 \
        --batch_size 4 --val_batch_size 4 --learning_rate 0.003 \
        --block_mode column --sampler grid \
        --focal_gamma 2.0 --oversample_rare 1.0 --aug_preset strong --cooldown_sec 0 \
        > "$4" 2>&1 &
    echo $!
}

logdir_of() { # $1 logfile — wait for train_model to announce its run dir
    local d=''
    for _ in $(seq 1 60); do
        d=$(grep -aoE "Logging to directory: logs/[0-9_]+" "$1" 2>/dev/null | head -1 | awk '{print $4}')
        [ -n "$d" ] && break
        sleep 5
    done
    echo "$d"
}

launch E1 64 1 v2_e1.log > /tmp/e1pid; P1=$(cat /tmp/e1pid); RUN1=$(logdir_of v2_e1.log)  # FIXME: launch in $() ran bg in a subshell — wait(1) saw no child (exit 127). Scoring reattached via run_v2_score_variants.sh
P2=$(launch E2 64 2 v2_e2.log); RUN2=$(logdir_of v2_e2.log)
echo "=== E1 (k64 c1) pid=$P1 -> $RUN1 | E2 (k64 c2) pid=$P2 -> $RUN2 ==="
wait "$P1"; S1=$?
wait "$P2"; S2=$?
echo "=== E1 exit=$S1  E2 exit=$S2  $(date '+%F %T') ==="

score() { # $1 tag  $2 rundir  $3 knn  $4 curves
    [ -n "$2" ] || { echo "=== [$1] NO RUN DIR ==="; return; }
    for W in best_model.pth final_model.pth; do
        [ -f "$2/${W}" ] || continue
        OUT="$2/test_full_${W%.pth}.json"
        if "$PYTHON" evaluate_full.py --model_weights "$2/${W}" \
                --config model_params_room.json --mode chunk --sampler grid \
                --block_size 20480 --core_max 12288 --halo 1.0 \
                --arch v2 --v2_knn "$3" --v2_curves "$4" \
                --enc_channels 64,192,320,448 --bottleneck_dim 256 \
                --out "$OUT"; then
            m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [$1] RESULT ${W}: mIoU=${m} (v1 baseline 58.82) run=$2 ==="
        else
            echo "=== [$1] EVAL FAILED ${W} ==="
        fi
    done
}
score E1 "$RUN1" 64 1
score E2 "$RUN2" 64 2
echo "=== VARIANTS COMPLETE $(date '+%F %T') ==="
