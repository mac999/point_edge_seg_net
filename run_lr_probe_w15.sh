#!/usr/bin/env bash
# ============================================================
#  LR probe for width_mult 1.5 (diagnostic, ~3 h total)
#  Why: at lr 0.003 / 60 ep the w1.5 model reaches only val 0.843
#  (w1.0: 0.930) with smooth curves and NO gradient clipping
#  (norms 0.3-5.3 < 8.0 threshold, measured) -- so the recipe, not
#  stability, is the suspect. Three 8-epoch runs bracket the LR:
#    0.0015 / 0.003 (reference) / 0.006
#  Compare epoch-8 train_loss & val_acc; the winner sets the LR for
#  the real 60-120 epoch width run. Cosine T_max=8 distorts all
#  three equally, so the comparison stays valid.
#  Usage: nohup ./run_lr_probe_w15.sh > lr_probe.log 2>&1 &
# ============================================================
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"

for LR in 0.0015 0.003 0.006; do
    echo "=== [PROBE lr=$LR] START $(date '+%F %T') ==="
    "$PYTHON" train_model.py \
        --config model_params.json \
        --processed_data_path ./processed_s3dis \
        --block_data_path ./block_s3dis \
        --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 \
        --test_area Area_5 \
        --num_epochs 8 \
        --batch_size 10 \
        --val_batch_size 18 \
        --learning_rate "$LR" \
        --block_size 8192 \
        --block_mode column \
        --column_window 2.0 \
        --column_stride 2.0 \
        --focal_gamma 2.0 \
        --width_mult 1.5 \
        --oversample_rare 1.0 \
        --no_wandb \
        --cooldown_sec 0 || { echo "=== [PROBE lr=$LR] FAILED ==="; continue; }
    run=$(ls -td logs/*/ | head -1)
    tail -1 "${run}training_log.csv" | awk -F, -v lr="$LR" -v r="$run" \
        '{printf "=== [PROBE lr=%s] EP8 train_loss=%s train_acc=%s val_acc=%s run=%s ===\n", lr, $2, $3, $5, r}'
done
echo "=== PROBE COMPLETE $(date '+%F %T') ==="
echo "Reference ep8 (60ep-schedule): EXP3b(w1.5,lr .003) train_acc=0.682 val=0.751 / EXP1(w1.0,lr .003) train_acc=0.820 val=0.771"
