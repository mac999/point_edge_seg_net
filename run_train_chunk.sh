#!/usr/bin/env bash
# ============================================================
#  VOXEL-CHUNK training: the large-cloud pipeline (S3DIS Area 5)
#
#  Pipeline (see voxel_chunk.py for the measurements behind each step):
#    voxelize 4 cm -> KD median split (XY only) to 12,288-point cores
#    -> 1 m halo of REAL neighbouring points -> 20,480-point fixed chunks
#    -> only core points are supervised (halo is context, never scored)
#
#  vs the 2 m block baseline (EXP1, mIoU 54.35):
#    input extent   2.0 x 2.0 m  ->  3.3 x 3.1 m   (2.6x the area)
#    density        2048 pts/m2  ->  ~1000 pts/m2  (DeLA uses 998)
#    coverage       subsample     ->  exact (every voxel scored once)
#    cost/epoch     1.00x         ->  1.76x
#
#  7D features (model_params_room.json): the stored spatial channels are not
#  rotation/scale invariant (measured MAE up to 67% of their own std), so they
#  are incompatible with the strong augmentation this recipe needs. Normals
#  co-rotate and curvature is eigenvalue-based, so 7D is augmentation-safe.
#  DeLA reaches 74.1 mIoU on just 4 channels.
#
#  --sampler grid: FPS is quadratic and would dominate at 20k points/chunk.
#  VRAM: 20,480 x batch 4 = 81,920 points = the same as the baseline's
#  8192 x 10, i.e. ~13.4 GB, well inside the 24 GB target.
#
#  Score with the MATCHING protocol:
#    python evaluate_full.py --model_weights logs/<run>/best_model.pth \
#        --config model_params_room.json --mode chunk --sampler grid \
#        --block_size 20480 --core_max 12288 --halo 1.0
# ============================================================

cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch. Activate the env or set PYTHON=." ; exit 1
fi

"$PYTHON" train_model.py \
    --config model_params_room.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./chunk_s3dis \
    --block_size 20480 \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 \
    --test_area Area_5 \
    --num_epochs 60 \
    --batch_size 4 \
    --val_batch_size 4 \
    --learning_rate 0.003 \
    --block_mode column \
    --sampler grid \
    --focal_gamma 2.0 \
    --oversample_rare 1.0 \
    --aug_preset strong \
    --cooldown_sec 0
status=$?
[ $status -ne 0 ] && { echo "[ERROR] Training failed (exit $status)."; exit $status; }
echo "[DONE] Results in ./logs/YYYYMMDD_HHMMSS/"
