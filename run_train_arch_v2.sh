#!/usr/bin/env bash
# ============================================================
#  EXP 4: architecture scale-up, CORRECTED RECIPE (S3DIS Area 5)
#  --width_mult 1.5   : channels 96/192/384/768 (5.7M -> 14.1M params)
#  --mid_transformer  : 1-layer attention at the ~860-pt mid level,
#                       ZERO-INIT residual+pos_enc (exact identity at
#                       init -- the non-identity version collapsed
#                       training, logs/20260723_091040)
#  --learning_rate 0.0015 : measured by LR probe (8-ep, w1.5):
#        lr 0.0015 -> train_acc 0.827 / val 0.847   <- winner
#        lr 0.003  -> train_acc 0.741 / val 0.746   (EXP3b's recipe)
#        lr 0.006  -> train_acc 0.710 / val 0.737
#    The wider net needs half the LR; at 0.003 it undertrained for
#    60 epochs (EXP3b full-protocol mIoU 46.90 vs EXP1 54.35).
#  + --oversample_rare 1.0 (same as EXP1 for comparability)
#  ~20.4 GB peak at batch 10 (DGX Spark OK), ~7.5-8 h for 60 epochs.
#  Score afterwards:
#    python evaluate_full.py --model_weights logs/<run>/best_model.pth \
#        --width_mult 1.5 --mid_transformer
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
    --learning_rate 0.0015 \
    --block_size 8192 \
    --block_mode column \
    --column_window 2.0 \
    --column_stride 2.0 \
    --focal_gamma 2.0 \
    --width_mult 1.5 \
    --mid_transformer \
    --oversample_rare 1.0 \
    --cooldown_sec 0
status=$?
[ $status -ne 0 ] && { echo "[ERROR] Training failed (exit $status)."; exit $status; }
echo "[DONE] Results in ./logs/YYYYMMDD_HHMMSS/"
