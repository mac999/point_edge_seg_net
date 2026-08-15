#!/usr/bin/env bash
# ============================================================
#  EXP 4b: WIDTH-ONLY, corrected LR (S3DIS Area 5 held-out)
#  --width_mult 1.5, NO --mid_transformer, lr 0.0015.
#  History: width 1.5 @ lr 0.003 undertrained (EXP3b, 46.90 mIoU);
#  LR probe picked 0.0015 (ep8 train_acc 0.827 vs 0.741); adding
#  the (identity-fixed) mid transformer STILL stalled optimization
#  (EXP4, logs/20260724_032732: val 0.828@59ep vs probe 0.847@8ep,
#  only difference = midT) -> midT abandoned. This is the last
#  arch candidate: pure width at the probed LR. Compare against
#  EXP1 (logs/20260722_201733, 54.35 full-protocol).
#  Score afterwards:
#    python evaluate_full.py --model_weights logs/<run>/best_model.pth --width_mult 1.5
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
    --oversample_rare 1.0 \
    --cooldown_sec 0
status=$?
[ $status -ne 0 ] && { echo "[ERROR] Training failed (exit $status)."; exit $status; }
echo "[DONE] Results in ./logs/YYYYMMDD_HHMMSS/"
