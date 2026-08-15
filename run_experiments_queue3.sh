#!/usr/bin/env bash
# ============================================================
#  Queue 3: EXP4 = width 1.5 + fixed midT + lr 0.0015 (probe winner)
#  -> full-protocol eval. Compare against EXP1 (54.35) / baseline (53.85).
#  Usage:  nohup ./run_experiments_queue3.sh > experiments_queue3.log 2>&1 &
# ============================================================
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }

echo "=== QUEUE3 START $(date '+%F %T') ==="
before=$(newest_run)
if bash run_train_arch_v2.sh; then
    run=$(newest_run)
    if [ -n "$run" ] && [ "$run" != "$before" ] && [ -f "${run}best_model.pth" ]; then
        echo "=== [EXP4-arch-lr0015] TRAIN DONE -> $run $(date '+%F %T') ==="
        if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth" --width_mult 1.5 --mid_transformer; then
            miou=$("$PYTHON" -c "import json;print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
            echo "=== [EXP4-arch-lr0015] RESULT full-protocol mIoU=${miou} (EXP1 54.35 / baseline 53.85) run=${run} $(date '+%F %T') ==="
        else
            echo "=== [EXP4-arch-lr0015] EVAL FAILED $(date '+%F %T') ==="
        fi
    else
        echo "=== [EXP4-arch-lr0015] TRAIN FAILED (no new run) $(date '+%F %T') ==="
    fi
else
    echo "=== [EXP4-arch-lr0015] TRAIN FAILED $(date '+%F %T') ==="
fi
echo "=== QUEUE3 COMPLETE $(date '+%F %T') ==="
