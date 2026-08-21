@echo off
rem ============================================================
rem  PointEdgeSegNet GLOBAL-context training (S3DIS Area 5 held-out)
rem  "global": --block_context appends a wide-area neighbourhood
rem  descriptor (verticality/horizontality/curvature/density +
rem  z-histogram, aggregated over a 4 m buffer around each block)
rem  to every block -> 10D base + 8D context = 18D input.
rem  Column blocks, Lovasz+Focal loss and curvature refresh are
rem  applied automatically. Reuses existing processed_s3dis
rem  (data_preparation.py does NOT need to be re-run: context is
rem  computed at block-build time). Context blocks are cached in
rem  block_s3dis_ctx, separate from the plain block_s3dis cache.
rem  NOTE: inference for the resulting model must also pass
rem        --block_context (python inference.py --block_context -m <model>).
rem ============================================================

cd /d "%~dp0"

python train_model.py ^
    --config model_params.json ^
    --processed_data_path ./processed_s3dis ^
    --block_data_path ./block_s3dis_ctx ^
    --train_areas Area_1 Area_2 Area_3 Area_4 Area_6 ^
    --test_area Area_5 ^
    --num_epochs 60 ^
    --batch_size 10 ^
    --val_batch_size 18 ^
    --learning_rate 0.003 ^
    --block_size 8192 ^
    --block_mode column ^
    --column_window 2.0 ^
    --column_stride 2.0 ^
    --focal_gamma 2.0 ^
    --block_context ^
    --cooldown_sec 0

rem To disable Weights & Biases logging, append:  --no_wandb
rem VRAM 8GB fallback: --batch_size 2 --block_size 4096
rem Plain (no-context) baseline: remove --block_context and set --block_data_path ./block_s3dis

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed. Check the output above.
) else (
    echo.
    echo [DONE] Training finished. Results are in .\logs\YYYYMMDD_HHMMSS\
)
pause
