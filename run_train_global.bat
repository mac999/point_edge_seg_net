@echo off
rem ============================================================
rem  PointEdgeSegNet v1.0 training (S3DIS Area 5 held-out)
rem  Improved options: context-preserving column blocks,
rem  Lovasz+Focal loss / curvature refresh are applied automatically.
rem  Reuses existing processed_s3dis (curvature channel is
rem  refreshed at block-build time; blocks cached in block_s3dis).
rem ============================================================

cd /d "%~dp0"

python train_model.py ^
    --config model_params.json ^
    --processed_data_path ./processed_s3dis ^
    --block_data_path ./block_s3dis ^
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
    --cooldown_sec 0

rem To disable Weights & Biases logging, append:  --no_wandb
rem VRAM 8GB fallback: --batch_size 2 --block_size 4096

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed. Check the output above.
) else (
    echo.
    echo [DONE] Training finished. Results are in .\logs\YYYYMMDD_HHMMSS\
)
pause
