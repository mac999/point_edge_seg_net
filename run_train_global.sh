#!/usr/bin/env bash
# ============================================================
#  PointEdgeSegNet GLOBAL-context training + FULL evaluation
#  (S3DIS Area 5 held-out, one command end-to-end)
#
#  1) TRAIN with --block_context: the 8D wide-area neighbourhood
#     descriptor per block (10D base + 8D context = 18D input).
#     Injection point is the model default --context_mode
#     'bottleneck' (zero-initialized residual): at init the network
#     is exactly the no-context baseline, so the worst case is
#     "context ignored". The legacy 'input' mode (constant channels
#     through the per-point path) measured -6.4 mIoU
#     (logs/20260722_104628) and is NOT used.
#     Context blocks are cached in block_s3dis_ctx (reused if present).
#
#  2) EVALUATE with evaluate_full.py: standard S3DIS protocol --
#     ALL Area 5 points (coverage-guaranteed blocks + per-point
#     voting), per-class acc/IoU + confusion matrix. This is the
#     number comparable to published results (paper baselines:
#     our 10D baseline 53.85 mIoU; PTv3 73.4). Results land in
#     logs/<run>/test_full_summary.json.
#
#  Inference for the resulting model must pass the SAME flags:
#     python inference.py --block_context --context_mode bottleneck -m <model>
# ============================================================

cd "$(dirname "$0")" || exit 1

PYTHON="${PYTHON:-python}"

if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "[ERROR] '$PYTHON' has no PyTorch."
    echo "        Activate the training env first (e.g. 'conda activate dev'),"
    echo "        or run:  PYTHON=/path/to/env/bin/python $0"
    exit 1
fi

# Remember the newest existing run so we can detect the one this training creates.
before_run=$(ls -td logs/*/ 2>/dev/null | head -1)

"$PYTHON" train_model.py \
    --config model_params.json \
    --processed_data_path ./processed_s3dis \
    --block_data_path ./block_s3dis_ctx \
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
    --block_context \
    --context_mode bottleneck \
    --cooldown_sec 0
status=$?

# To disable Weights & Biases logging, append:  --no_wandb
# Rare-class oversampling (sofa/board/column):  --oversample_rare 1.0
# Small-GPU fallback (8 GB):                    --batch_size 2 --block_size 4096

if [ $status -ne 0 ]; then
    echo
    echo "[ERROR] Training failed (exit $status). Check the output above."
    exit $status
fi

run=$(ls -td logs/*/ 2>/dev/null | head -1)
if [ -z "$run" ] || [ "$run" = "$before_run" ] || [ ! -f "${run}best_model.pth" ]; then
    echo "[ERROR] Training reported success but no new logs/<run>/best_model.pth was found."
    exit 1
fi
echo
echo "[TRAIN DONE] $run  -- starting full-protocol evaluation (~30 min on Area 5)..."
echo

"$PYTHON" evaluate_full.py \
    --model_weights "${run}best_model.pth" \
    --block_context \
    --context_mode bottleneck
eval_status=$?

if [ $eval_status -ne 0 ]; then
    echo
    echo "[ERROR] Full evaluation failed (exit $eval_status). The trained model is intact at:"
    echo "        ${run}best_model.pth"
    echo "        Re-run manually:  $PYTHON evaluate_full.py --model_weights ${run}best_model.pth --block_context --context_mode bottleneck"
    exit $eval_status
fi

miou=$("$PYTHON" -c "import json; print(f\"{json.load(open('${run}test_full_summary.json'))['overall_metrics']['mIoU']*100:.2f}\")" 2>/dev/null)
echo
echo "[DONE] Training + full evaluation complete."
echo "       Run dir             : ${run}"
echo "       Full-protocol mIoU  : ${miou}  (10D baseline: 53.85)"
echo "       Details             : ${run}test_full_summary.json (per-class acc/IoU + confusion matrix)"
