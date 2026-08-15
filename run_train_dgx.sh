#!/usr/bin/env bash
# ============================================================
#  PointEdgeSegNet WIDE-CONTEXT training (S3DIS Area 5 held-out)
#  Tuned for NVIDIA DGX Spark (GB10, 124 GB unified memory).
#
#  WHAT THIS DOES
#  Widens the column 2.0 m -> 3.0 m and scales block_size by the
#  same area ratio (8192 * 2.25 = 18432), so point density is
#  unchanged while each point's receptive field grows 2.25x.
#  Peak memory ~30 GB - impossible on a 24 GB card, cheap here.
#  Stride stays at 1.5 m (50% overlap), NOT 3.0 m: see below.
#
#  WHY stride 1.5 AND NOT 3.0  (this is the whole point of this file)
#  A first attempt used --column_window 3.0 --column_stride 3.0 and
#  it LOST accuracy (logs/20260721_233224: OA 84.22 / mAcc 59.26 /
#  mIoU 52.10, vs 85.53 / 61.77 / 53.44 for the 2.0 m baseline).
#  Cause: widening window and stride together collapses the block
#  count, so training blocks fell 1400 -> 684 (-51%) and steps per
#  epoch 140 -> 68. Points per epoch actually went UP 10%, but SGD
#  learns from distinct samples, not from points, and the model
#  simply overfit: train_loss 0.159 (lower) with val_loss 0.447
#  (higher) vs 0.178 / 0.411 for the baseline.
#  Holding stride at window/2 keeps ~1225 training blocks (88% of
#  baseline) AND gives 50% block overlap, which acts as extra
#  augmentation. Cost: ~2.0x baseline points per epoch, so expect
#  ~10 min/epoch (~10 h for 60 epochs) on this box.
#
#  CHEAPER ALTERNATIVE, TRY IT FIRST
#  run_train_global.sh (--block_context) adds wide-area context as
#  12 extra input channels while keeping window 2.0 / stride 2.0,
#  so the block count - and therefore the sample count - is
#  untouched. It costs roughly baseline time and cannot hit the
#  failure mode described above. It has never actually been run.
#
#  IMPORTANT: --block_data_path MUST be a fresh folder. The block
#  cache is never invalidated (preprocess_dataset_columns() in
#  train_model.py returns early if the folder holds any .pt file,
#  regardless of window/stride/block_size), so pointing this at
#  ./block_s3dis or ./block_s3dis_w3 would silently train on the
#  wrong geometry.
#  Inference must use the SAME --block_size / window / stride.
#
#  NOTE ON COMPARING RUNS: changing window/block_size also
#  re-samples the Area 5 test set (696 blocks/5.63M pts at 2.0 m vs
#  379/6.80M at 3.0 m), so test numbers across configs are not
#  strictly like-for-like. Treat differences under ~1 point as noise.
# ============================================================

cd "$(dirname "$0")" || exit 1

PYTHON="${PYTHON:-python}"

# Fail early and clearly instead of dying inside train_model.py.
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch."
    echo "        Activate the training env first (e.g. 'conda activate dev'),"
    echo "        or run:  PYTHON=/path/to/env/bin/python $0"
    exit 1
fi

"$PYTHON" train_model.py \
    --config model_params.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./block_s3dis_w3s15 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 \
    --test_area Area_5 \
    --num_epochs 60 \
    --batch_size 10 \
    --val_batch_size 18 \
    --learning_rate 0.003 \
    --block_size 18432 \
    --block_mode column \
    --column_window 3.0 \
    --column_stride 1.5 \
    --focal_gamma 2.0 \
    --cooldown_sec 0
status=$?

# --num_epochs: 60 completes the cosine schedule in ~10 h here. Raising it is safe now that
#   early stopping waits for the LR to anneal (train_model.py), but scale the time budget:
#   80 epochs ~= 13.5 h.
# To disable Weights & Biases logging, append:  --no_wandb
# Long run: nohup ./run_train_dgx.sh > train_dgx.log 2>&1 &

if [ $status -ne 0 ]; then
    echo
    echo "[ERROR] Training failed (exit $status). Check the output above."
    exit $status
fi

echo
echo "[DONE] Training finished. Results are in ./logs/YYYYMMDD_HHMMSS/"
