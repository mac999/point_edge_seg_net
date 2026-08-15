#!/usr/bin/env bash
# Stage E: wait for any running training to finish, then train room mode and score it.
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }
# Serialize behind any training/eval already on the GPU so they never overlap.
#
# `pgrep -f train_model.py` is NOT usable here: it also matches any shell whose command
# line merely CONTAINS the string -- including the monitoring commands used to inspect
# this very queue. That false match stalled an earlier run of this script for two hours
# with an idle GPU. Resolve each candidate PID to its real executable and count only
# python ones.
gpu_job_running() {
    local pid exe
    for pid in $(pgrep -f 'train_model\.py|evaluate_full\.py' 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || continue
        case "$exe" in *python*) return 0 ;; esac
    done
    return 1
}
while gpu_job_running; do sleep 60; done
echo "=== STAGE-E START $(date '+%F %T') ==="
before=$(newest_run)
if bash run_train_stageE.sh; then
    run=$(newest_run)
    if [ -n "$run" ] && [ "$run" != "$before" ] && [ -f "${run}best_model.pth" ]; then
        echo "=== [STAGE-E] TRAIN DONE -> $run $(date '+%F %T') ==="
        if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth" --config model_params_room.json --mode room --sampler grid; then
            miou=$("$PYTHON" -c "import json;print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [STAGE-E] RESULT full-protocol mIoU=${miou} (EXP1 54.35 / baseline 53.85) run=${run} $(date '+%F %T') ==="
        else
            echo "=== [STAGE-E] EVAL FAILED $(date '+%F %T') ==="
        fi
    else
        echo "=== [STAGE-E] TRAIN FAILED (no new run) $(date '+%F %T') ==="
    fi
else
    echo "=== [STAGE-E] TRAIN FAILED $(date '+%F %T') ==="
fi
echo "=== STAGE-E COMPLETE $(date '+%F %T') ==="
