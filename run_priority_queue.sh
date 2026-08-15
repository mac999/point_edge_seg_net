#!/usr/bin/env bash
# ============================================================
#  Priority queue, ordered by expected value:
#    P1  50% overlap scoring of EXP1        (no training; lit. +2.3~2.7)
#    P2  EXP1 config at 150 epochs          (2.5x iterations) + scoring
#    P3  P2's best checkpoint with TTA      (no training; lit. +0.5~1)
#  Each step logs a RESULT line; a failure logs and the queue moves on.
#  Usage: nohup ./run_priority_queue.sh > priority.log 2>&1 &
# ============================================================
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
BEST=logs/20260722_201733/best_model.pth      # EXP1, full-protocol mIoU 54.35
newest_run() { ls -td logs/*/ 2>/dev/null | head -1; }

# Serialize only against OUR OWN jobs. Two filters, both learned the hard way:
#   1. the process must really be python -- a shell whose command line merely mentions
#      the script name is not a job (that false match idled the GPU for two hours);
#   2. its working directory must be THIS directory -- another checkout of this project
#      trains concurrently on the same GPU, and blocking on it stalls us indefinitely.
#      The GPU has ample memory (121 GB) so sharing it costs speed, not correctness.
HERE=$(pwd -P)
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
wait_for_gpu() { while gpu_job_running; do sleep 60; done; }

miou_of() { "$PYTHON" -c "import json;print(f\"{json.load(open('$1'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null; }

echo "=== PRIORITY QUEUE START $(date '+%F %T') ==="

# ---------- P1: 50% overlap scoring (no retraining) ----------
wait_for_gpu
echo "=== [P1 overlap] START $(date '+%F %T') ==="
OUT1=logs/20260722_201733/test_full_summary_stride1.json
if "$PYTHON" evaluate_full.py --model_weights "$BEST" --stride 1.0 --out "$OUT1"; then
    echo "=== [P1 overlap] RESULT mIoU=$(miou_of $OUT1) (EXP1 no-overlap 54.35) $(date '+%F %T') ==="
else
    echo "=== [P1 overlap] FAILED $(date '+%F %T') ==="
fi

# ---------- P2: 2.5x longer schedule ----------
wait_for_gpu
echo "=== [P2 long-150ep] START $(date '+%F %T') ==="
before=$(newest_run)
if bash run_train_long.sh; then
    run=$(newest_run)
    if [ -n "$run" ] && [ "$run" != "$before" ] && [ -f "${run}best_model.pth" ]; then
        echo "=== [P2 long-150ep] TRAIN DONE -> $run $(date '+%F %T') ==="
        if "$PYTHON" evaluate_full.py --model_weights "${run}best_model.pth"; then
            echo "=== [P2 long-150ep] RESULT mIoU=$(miou_of ${run}test_full_summary.json) (EXP1 54.35) $(date '+%F %T') ==="
            LONG_BEST="${run}best_model.pth"; LONG_DIR="$run"
        else
            echo "=== [P2 long-150ep] EVAL FAILED $(date '+%F %T') ==="
        fi
    else
        echo "=== [P2 long-150ep] TRAIN FAILED (no new run) $(date '+%F %T') ==="
    fi
else
    echo "=== [P2 long-150ep] TRAIN FAILED $(date '+%F %T') ==="
fi

# ---------- P3: TTA on whichever checkpoint is better ----------
wait_for_gpu
CKPT="${LONG_BEST:-$BEST}"; DIR="${LONG_DIR:-logs/20260722_201733/}"
echo "=== [P3 tta] START on $CKPT $(date '+%F %T') ==="
OUT3="${DIR}test_full_summary_tta.json"
if "$PYTHON" evaluate_full.py --model_weights "$CKPT" --tta 5 --tta_flip --out "$OUT3"; then
    echo "=== [P3 tta] RESULT mIoU=$(miou_of $OUT3) (10 views) $(date '+%F %T') ==="
else
    echo "=== [P3 tta] FAILED $(date '+%F %T') ==="
fi

echo "=== PRIORITY QUEUE COMPLETE $(date '+%F %T') ==="
