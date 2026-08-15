#!/usr/bin/env bash
# Voxel-chunk experiment: wait for the cache + our GPU, train, then score with --mode chunk.
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
HERE=$(pwd -P)
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
# wait for the chunk cache to finish building
while pgrep -f "prepare_chunk_cache" > /dev/null; do sleep 30; done
# and behind the 150-epoch run so the two experiments never share the GPU
while gpu_job_running; do sleep 60; done
echo "=== CHUNKINV START $(date '+%F %T') ==="
before=$(newest_run)
if bash run_train_chunk_inv.sh; then
    run=$(newest_run)
    if [ -n "$run" ] && [ "$run" != "$before" ] && [ -f "${run}best_model.pth" ]; then
        echo "=== [CHUNKINV] TRAIN DONE -> $run $(date '+%F %T') ==="
        if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth" \
              --config model_params.json --mode chunk --sampler grid \
              --block_size 20480 --core_max 12288 --halo 1.0; then
            miou=$("$PYTHON" -c "import json;print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [CHUNKINV] RESULT full-protocol mIoU=${miou} (best so far 56.45) run=${run} $(date '+%F %T') ==="
        else
            echo "=== [CHUNKINV] EVAL FAILED $(date '+%F %T') ==="
        fi
    else
        echo "=== [CHUNKINV] TRAIN FAILED (no new run) $(date '+%F %T') ==="
    fi
else
    echo "=== [CHUNKINV] TRAIN FAILED $(date '+%F %T') ==="
fi
echo "=== CHUNKINV COMPLETE $(date '+%F %T') ==="
