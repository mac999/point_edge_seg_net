#!/usr/bin/env bash
# Scoring stage for the E1/E2 variant runs, reattached after run_v2_variants.sh's
# wait bug (launch() in $() command substitution → background PID not a child of the
# queue shell → `wait` returned 127 and scoring was skipped; trainings were unharmed).
# Polls the orphaned PIDs with kill -0, then scores both runs.
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-/home/tom3/miniconda3/envs/pesn/bin/python}"
P1=879069; RUN1=logs/20260815_110116   # E1: k64 curves=1
P2=879124; RUN2=logs/20260815_110121   # E2: k64 curves=2

echo "=== SCORE-VARIANTS: waiting for pids $P1 $P2 $(date '+%F %T') ==="
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do sleep 60; done
echo "=== SCORE-VARIANTS: trainings ended $(date '+%F %T') ==="

score() { # $1 tag  $2 rundir  $3 knn  $4 curves
    for W in best_model.pth final_model.pth; do
        [ -f "$2/${W}" ] || { echo "=== [$1] missing $2/${W} ==="; continue; }
        OUT="$2/test_full_${W%.pth}.json"
        if "$PYTHON" evaluate_full.py --model_weights "$2/${W}" \
                --config model_params_room.json --mode chunk --sampler grid \
                --block_size 20480 --core_max 12288 --halo 1.0 \
                --arch v2 --v2_knn "$3" --v2_curves "$4" \
                --enc_channels 64,192,320,448 --bottleneck_dim 256 \
                --out "$OUT"; then
            m=$("$PYTHON" -c "import json;print(f\"{json.load(open('$OUT'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [$1] RESULT ${W}: mIoU=${m} (v1 58.82 | v2-k32 47.62) run=$2 ==="
        else
            echo "=== [$1] EVAL FAILED ${W} ==="
        fi
    done
}
score E1 "$RUN1" 64 1
score E2 "$RUN2" 64 2
echo "=== SCORE-VARIANTS COMPLETE $(date '+%F %T') ==="
