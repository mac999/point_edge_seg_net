#!/usr/bin/env bash
# ============================================================
#  LONGER SCHEDULE: EXP1's exact configuration, 60 -> 150 epochs
#
#  WHY: training length is the largest untested gap against the closest
#  reference model. EXP1 does 60 epochs x 140 steps = 8,400 optimizer
#  iterations; DeLA (7.0M params, plain kNN k=24, 74.1 mIoU on this
#  benchmark) does ~76,500. Pointcept's S3DIS recipes are the same order
#  (100 scheduler epochs x loop=30). 150 epochs = 21,000 iterations, i.e.
#  2.5x ours -- still well short of the references, but enough to show
#  whether the direction pays.
#
#  Everything else is EXP1 verbatim (10D features, legacy augmentation,
#  --oversample_rare 1.0, block mode 2 m / 8192, lr 0.003) so the only
#  variable is schedule length.
#
#  Safe to lengthen now: early stopping refuses to fire until the cosine
#  has annealed to its floor (train_model.py EARLY_STOP_REQUIRE_ANNEAL),
#  so a longer T_max is no longer silently truncated -- the failure that
#  wasted logs/20260721_233224.
#
#  VRAM: ~13.4 GB (unchanged from EXP1), well inside the 24 GB target.
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
    --num_epochs 150 \
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
