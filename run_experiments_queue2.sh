#!/usr/bin/env bash
# ============================================================
#  Follow-up ablation queue (after EXP3's collapse):
#    EXP3b run_train_arch_v2b.sh (width 1.5 only)        -> evaluate_full --width_mult 1.5
#    EXP3c run_train_arch_v2.sh  (width 1.5 + FIXED midT: -> evaluate_full --width_mult 1.5 --mid_transformer
#          zero-init residual+pos_enc, exact identity at init)
#  Attribution (full-protocol mIoU): EXP3b - EXP1(54.35) = width effect;
#  EXP3c - EXP3b = (fixed) mid-transformer effect.
#  Usage:  nohup ./run_experiments_queue2.sh > experiments_queue2.log 2>&1 &
# ============================================================

cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"

newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }

run_exp() {
    local name="$1" train_script="$2"; shift 2
    echo "=== [$name] TRAIN START $(date '+%F %T') ==="
    local before; before=$(newest_run)
    if ! bash "$train_script"; then
        echo "=== [$name] TRAIN FAILED $(date '+%F %T') ==="
        return 1
    fi
    local run; run=$(newest_run)
    if [ -z "$run" ] || [ "$run" = "$before" ] || [ ! -f "$run/best_model.pth" ]; then
        echo "=== [$name] TRAIN FAILED (no new run dir/best_model) $(date '+%F %T') ==="
        return 1
    fi
    echo "=== [$name] TRAIN DONE -> $run $(date '+%F %T') ==="
    if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth" "$@"; then
        local miou
        miou=$("$PYTHON" -c "import json;print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
        echo "=== [$name] RESULT full-protocol mIoU=${miou} (EXP1 54.35 / baseline 53.85) run=${run} $(date '+%F %T') ==="
    else
        echo "=== [$name] EVAL FAILED $(date '+%F %T') ==="
    fi
}

echo "=== QUEUE2 START $(date '+%F %T') (python=$PYTHON) ==="
run_exp "EXP3b-width-only" run_train_arch_v2b.sh --width_mult 1.5
run_exp "EXP3c-width+fixed-midT" run_train_arch_v2.sh --width_mult 1.5 --mid_transformer
echo "=== QUEUE2 COMPLETE $(date '+%F %T') ==="
