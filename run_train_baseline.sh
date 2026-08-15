#!/usr/bin/env bash
# ============================================================
#  PointEdgeSegNet BASELINE training (S3DIS Area 5 held-out)
#  Plain 10D features (NO --block_context), 2.0 m columns,
#  8192-point blocks, 60 epochs.
#  Reuses the 10D block cache in block_s3dis/.
#  Context counterpart:   run_train_global.sh  (22D, block_s3dis_ctx)
#  Wide-context / DGX:    run_train_dgx.sh     (3.0 m columns, block_s3dis_w3)
#
#  NOTE: the block cache is never invalidated (train_model.py:514
#  returns early when the folder already holds .pt files), so the
#  numbers this produces are only comparable to another run that
#  used the SAME block_s3dis/ contents.
# ============================================================

cd "$(dirname "$0")" || exit 1

PYTHON="${PYTHON:-python}"

if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch."
    echo "        Activate the training env first (e.g. 'conda activate dev'),"
    echo "        or run:  PYTHON=/path/to/env/bin/python $0"
    exit 1
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
    --cooldown_sec 0
status=$?

# To disable Weights & Biases logging, append:  --no_wandb
# Small-GPU fallback (8 GB):                    --batch_size 2 --block_size 4096

if [ $status -ne 0 ]; then
    echo
    echo "[ERROR] Training failed (exit $status). Check the output above."
    exit $status
fi

echo
echo "[DONE] Training finished. Results are in ./logs/YYYYMMDD_HHMMSS/"
