#!/usr/bin/env bash
# ============================================================
#  STAGE A: augmentation normalization (S3DIS Area 5 held-out)
#
#  Identical to EXP1 (logs/20260722_201733, full-protocol mIoU 54.35)
#  except --aug_preset strong. Attribution: this run - 54.35 = the
#  augmentation effect.
#
#  WHY (measured):
#   - the legacy schedule DECAYS to prob 0.1 / strength 0.2 for the
#     last half of training; mean (prob x strength) = 0.046 over 60
#     epochs, ~20x below the constant full-strength augmentation every
#     S3DIS SOTA recipe uses;
#   - the val->test gap was a CONSTANT 8.4 points across baseline,
#     +block-context and width-1.5 runs -- i.e. generalization, not
#     capacity, is the binding constraint, and no architecture change
#     moved it;
#   - 'strong' adds full 360 deg yaw (legacy reached only +-36 deg late
#     in training), isotropic scale 0.8-1.2, mirroring p=0.5, small
#     tilts, colour drop p=0.2 and chromatic auto-contrast p=0.2.
#
#  EPOCHS 80 (not 60): stronger augmentation needs more iterations to
#  converge. The epoch-60 row of training_log.csv still gives a rough
#  same-length comparison, though the cosine T_max differs.
#  VRAM: unchanged from EXP1, ~13.4 GB peak (well under the 24 GB target).
#
#  Score afterwards (already wired into this script):
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
    --num_epochs 80 \
    --batch_size 10 \
    --val_batch_size 18 \
    --learning_rate 0.003 \
    --block_size 8192 \
    --block_mode column \
    --column_window 2.0 \
    --column_stride 2.0 \
    --focal_gamma 2.0 \
    --oversample_rare 1.0 \
    --aug_preset strong \
    --cooldown_sec 0
status=$?
[ $status -ne 0 ] && { echo "[ERROR] Training failed (exit $status)."; exit $status; }
echo "[DONE] Results in ./logs/YYYYMMDD_HHMMSS/"
