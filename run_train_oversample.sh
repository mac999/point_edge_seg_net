#!/usr/bin/env bash
# ============================================================
#  EXP 1: rare-class block oversampling (S3DIS Area 5 held-out)
#  Baseline 10D configuration + --oversample_rare 1.0.
#  Measured motivation: sofa appears in only 61/1400 training
#  blocks (4.4%); with power 1.0 its draw probability becomes
#  9.2% (x2.1), board x1.44, column x1.17, door unchanged.
#  Epoch length and all other settings identical to the
#  full-protocol baseline (logs/20260721_132243, mIoU 53.85) --
#  score afterwards with:
#    python evaluate_full.py --model_weights logs/<run>/best_model.pth
# ============================================================

cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch. Activate the env or set PYTHON=." ; exit 1
fi

"$PYTHON" train_model.py \
    --config model_params.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./block_s3dis \
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
    --oversample_rare 1.0 \
    --cooldown_sec 0
status=$?
[ $status -ne 0 ] && { echo "[ERROR] Training failed (exit $status)."; exit $status; }
echo "[DONE] Results in ./logs/YYYYMMDD_HHMMSS/"
