#!/usr/bin/env bash
# ============================================================
#  Sequential experiment queue for DGX Spark (one GPU, ~20 h):
#    EXP1 run_train_oversample.sh      (~5.5 h) -> evaluate_full
#    EXP2 run_train_ctx_bottleneck.sh  (~5.5 h) -> evaluate_full --block_context --context_mode bottleneck
#    EXP3 run_train_arch_v2.sh         (~7.5 h) -> evaluate_full --width_mult 1.5 --mid_transformer
#  Attribution chain (full-protocol mIoU, baseline = 53.85):
#    EXP1 - baseline  = oversampling effect
#    EXP2 - baseline  = bottleneck-context effect (zero-init: low risk)
#    EXP3 - EXP1      = architecture (width 1.5 + mid transformer) effect
#  A failed experiment logs [FAILED] and the queue moves on.
#  Usage:  nohup ./run_experiments_queue.sh > experiments_queue.log 2>&1 &
# ============================================================

cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"

newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }

run_exp() {
    local name="$1" train_script="$2"; shift 2   # rest = evaluate_full extra flags
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
    echo "=== [$name] FULL EVAL START ==="
    if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth" "$@"; then
        local miou
        miou=$("$PYTHON" -c "import json;print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
        echo "=== [$name] RESULT full-protocol mIoU=${miou} (baseline 53.85) run=${run} $(date '+%F %T') ==="
    else
        echo "=== [$name] EVAL FAILED $(date '+%F %T') ==="
    fi
}

echo "=== QUEUE START $(date '+%F %T') (python=$PYTHON) ==="
run_exp "EXP1-oversample"     run_train_oversample.sh
run_exp "EXP2-ctx-bottleneck" run_train_ctx_bottleneck.sh --block_context --context_mode bottleneck
run_exp "EXP3-arch-v2"        run_train_arch_v2.sh --width_mult 1.5 --mid_transformer
echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
