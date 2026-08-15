#!/usr/bin/env bash
# ============================================================
#  EXP 2: block-context via BOTTLENECK injection (S3DIS Area 5)
#  Replaces the failed input-mode injection (logs/20260722_104628,
#  mIoU -6.4): the 8D context descriptor now enters as a
#  ZERO-INITIALIZED residual at the bottleneck, so at init the
#  network is exactly the baseline and context can only be used
#  where it lowers the loss. Worst case = context ignored.
#  Reuses the existing 18D block cache (block_s3dis_ctx).
#  Score afterwards with:
#    python evaluate_full.py --model_weights logs/<run>/best_model.pth \
#        --block_context --context_mode bottleneck
# ============================================================

cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch. Activate the env or set PYTHON=." ; exit 1
fi

"$PYTHON" train_model.py \
    --config model_params.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./block_s3dis_ctx \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 \
    --test_area Area_5 \
    --num_epochs 60 \
    --batch_size 10 \
    --val_batch_size 18 \
    --learning_rate 0.003 \
    --block_size 8192 \
    --block_mode column \
    --column_window 2.0 \
    --column_stride 2.0 \
    --focal_gamma 2.0 \
    --block_context \
    --context_mode bottleneck \
    --cooldown_sec 0
status=$?
[ $status -ne 0 ] && { echo "[ERROR] Training failed (exit $status)."; exit $status; }
echo "[DONE] Results in ./logs/YYYYMMDD_HHMMSS/"
