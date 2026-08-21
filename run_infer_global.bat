@echo off
rem ============================================================
rem  PointEdgeSegNet GLOBAL-context inference
rem  Pairs with run_train_global.bat: --block_context appends the
rem  same wide-area neighbourhood descriptor (12D) per block, so
rem  the input matches an 18D context-trained model. Using a plain
rem  (10D, e.g. v1.1) model here fails fast with a dimension error.
rem
rem  Usage:
rem    run_infer_global.bat <model.pth> [input_cloud.txt]
rem      %1  path to a --block_context-trained best_model.pth (required)
rem      %2  input point cloud X Y Z [R G B] (optional; sample if omitted)
rem
rem  Output: <input>_segmented.las (colored, class in 'classification')
rem          and <input>_segmented.txt, next to the input file.
rem ============================================================

cd /d "%~dp0"

if "%~1"=="" (
    echo [ERROR] Model path required.
    echo Usage:  run_infer_global.bat logs\YYYYMMDD_HHMMSS\best_model.pth [input_cloud.txt]
    echo         The model must have been trained with --block_context ^(run_train_global.bat^).
    pause
    exit /b 1
)

set MODEL=%~1
set INPUT=%~2
if "%INPUT%"=="" set INPUT=./sample/area_6_conferenceRoom_1.txt

python inference.py ^
    --config model_params.json ^
    --model_weights "%MODEL%" ^
    --input_cloud "%INPUT%" ^
    --block_context ^
    --no_visualization

rem Extra accuracy (slower): append  --tta
rem Ensemble a second context-trained model: append  --ensemble path\to\other_best_model.pth

if errorlevel 1 (
    echo.
    echo [ERROR] Inference failed. Check the output above.
    echo Hint: a size-mismatch error means the model was NOT trained with --block_context.
) else (
    echo.
    echo [DONE] Segmentation LAS/TXT written next to: %INPUT%
)
pause
