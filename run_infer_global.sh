#!/usr/bin/env bash
# ============================================================
#  PointEdgeSegNet GLOBAL-context inference
#  Pairs with run_train_global.sh: --block_context appends the
#  same wide-area neighbourhood descriptor (12D) per block, so
#  the input matches an 18D context-trained model. Using a plain
#  (10D, e.g. v1.1) model here fails fast with a dimension error.
#
#  Usage:
#    ./run_infer_global.sh <model.pth> [input_cloud.txt]
#      $1  path to a --block_context-trained best_model.pth (required)
#      $2  input point cloud X Y Z [R G B] (optional; sample if omitted)
#
#  Output: <input>_segmented.las (colored, class in 'classification')
#          and <input>_segmented.txt, next to the input file.
# ============================================================

cd "$(dirname "$0")" || exit 1

if [ -z "${1:-}" ]; then
    echo "[ERROR] Model path required."
    echo "Usage:  ./run_infer_global.sh logs/YYYYMMDD_HHMMSS/best_model.pth [input_cloud.txt]"
    echo "        The model must have been trained with --block_context (run_train_global.sh)."
    exit 1
fi

MODEL="$1"
INPUT="${2:-./sample/area_6_conferenceRoom_1.txt}"

PYTHON="${PYTHON:-python}"

if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch."
    echo "        Activate the env first (e.g. 'conda activate dev'),"
    echo "        or run:  PYTHON=/path/to/env/bin/python $0 $*"
    exit 1
fi

"$PYTHON" inference.py \
    --config model_params.json \
    --model_weights "$MODEL" \
    --input_cloud "$INPUT" \
    --block_context \
    --no_visualization
status=$?

# Extra accuracy (slower): append  --tta
# Ensemble a second context-trained model: append  --ensemble path/to/other_best_model.pth

if [ $status -ne 0 ]; then
    echo
    echo "[ERROR] Inference failed (exit $status). Check the output above."
    echo "Hint: a size-mismatch error means the model was NOT trained with --block_context."
    exit $status
fi

echo
echo "[DONE] Segmentation LAS/TXT written next to: $INPUT"
