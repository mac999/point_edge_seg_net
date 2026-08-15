#!/usr/bin/env bash
# Stage A: train with strong augmentation, then score with the standard full protocol.
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }
echo "=== STAGE-A START $(date '+%F %T') ==="
before=$(newest_run)
if bash run_train_stageA.sh; then
    run=$(newest_run)
    if [ -n "$run" ] && [ "$run" != "$before" ] && [ -f "${run}best_model.pth" ]; then
        echo "=== [STAGE-A] TRAIN DONE -> $run $(date '+%F %T') ==="
        if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth"; then
            miou=$("$PYTHON" -c "import json;print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [STAGE-A] RESULT full-protocol mIoU=${miou} (EXP1 54.35 / baseline 53.85) run=${run} $(date '+%F %T') ==="
        else
            echo "=== [STAGE-A] EVAL FAILED $(date '+%F %T') ==="
        fi
    else
        echo "=== [STAGE-A] TRAIN FAILED (no new run) $(date '+%F %T') ==="
    fi
else
    echo "=== [STAGE-A] TRAIN FAILED $(date '+%F %T') ==="
fi
echo "=== STAGE-A COMPLETE $(date '+%F %T') ==="
