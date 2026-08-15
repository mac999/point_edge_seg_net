#!/usr/bin/env bash
# ============================================================
#  STAGE E: whole-room training (S3DIS Area 5 held-out)
#
#  The receptive-field fix. Instead of 2 m x 2 m columns of 8192
#  full-density points, train on whole rooms voxel-subsampled to
#  4 cm and center-cropped to 30,000 points, and TEST on the whole
#  room -- the protocol every S3DIS SOTA model uses.
#
#  WHY (measured on this repo):
#   - the block receptive field is hard-capped at 2.0 m, and the
#     classes that need more collapse into `wall` (column 77.7%,
#     door 42.4%, board 28.0% of their GT points);
#   - 4 cm voxelization buys 9x the area for 4.1x the points, and a
#     whole room is only 47-89k points;
#   - FPS made this impossible (24.1 s at 131k points); grid
#     subsampling is 54 ms, 446x faster and flat in N -- hence
#     --sampler grid, which is REQUIRED here.
#
#  Reference point: DeLA (arXiv 2308.16532) reaches 74.1 mIoU with
#  7.0M params and plain kNN k=24 using exactly this data recipe,
#  vs our 5.73M params at 54.35 with 2 m blocks.
#
#  MEASURED COST (bs 4, 30k points/room): ~17.7 GB peak -- inside the
#  24 GB target -- and 168 steps/epoch at loop=4.
#
#  FEATURES: model_params_room.json sets use_spatial=false -> 7D
#  [normals(3), curvature(1), rgb(3)]. This is REQUIRED for valid strong
#  augmentation: the stored spatial channels (density / angular anisotropy
#  / local structure) are computed once at preprocessing in the original
#  orientation and are NOT invariant to rotation or scaling -- measured MAE
#  0.077 / 0.162 / 0.145 against their own std of 0.240 / 0.242 / 0.300,
#  i.e. up to 67% of the feature's range. Under the legacy schedule only
#  ~10% of samples were rotated so this stayed diluted; at full strength it
#  corrupts every sample, which is the mechanism behind Stage A's
#  regression (subsampled mIoU 52.12 vs EXP1 54.05). Normals co-rotate and
#  curvature is eigenvalue-based, so the remaining 7D is augmentation-safe.
#  For reference DeLA reaches 74.1 mIoU on just 4 channels (rgb + height).
#
#  Score afterwards with the matching protocol (room mode + grid):
#    python evaluate_full.py --model_weights logs/<run>/best_model.pth \
#        --config model_params_room.json --mode room --sampler grid
# ============================================================

cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch. Activate the env or set PYTHON=." ; exit 1
fi

"$PYTHON" train_model.py \
    --config model_params_room.json \
    --processed_data_path ./processed_s3dis \
    --block_mode room \
    --room_data_path ./room_s3dis \
    --room_grid 0.04 \
    --room_max_points 30000 \
    --room_loop 4 \
    --sampler grid \
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 \
    --test_area Area_5 \
    --num_epochs 60 \
    --batch_size 4 \
    --val_batch_size 4 \
    --learning_rate 0.003 \
    --focal_gamma 2.0 \
    --aug_preset strong \
    --cooldown_sec 0
status=$?
[ $status -ne 0 ] && { echo "[ERROR] Training failed (exit $status)."; exit $status; }
echo "[DONE] Results in ./logs/YYYYMMDD_HHMMSS/"
