@echo off
rem ============================================================
rem  The released SemanticBridge run: the 12 m / 4 cm recipe trained
rem  on every labelled block of the 15 training bridges.
rem
rem  The context sweep scales the voxel with the window and peaks at
rem  6 m. Two controls that change one of them at a time show that
rem  peak is the coarser voxel's: at a fixed 4 cm voxel the 12 m
rem  window scores 70.82 single-view against 69.81.
rem
rem  Validation blocks are folded back into training, 400 epochs are
rem  fixed in advance and early stopping is disabled, so the printed
rem  validation numbers are IN-SAMPLE. Report final_model.pth.
rem
rem  Measured (84,153,822 points, official 15/5 split)
rem     single          mIoU 70.80   OA 92.44   ^<- protocol matches the baselines
rem     overlap_mirror  mIoU 71.87   OA 92.75   ^<- votes; baselines do not
rem
rem  Usage
rem    run_bridge_w12_v4_all.bat            score the released checkpoint
rem    run_bridge_w12_v4_all.bat --train    train from scratch first (~4 days on one card)
rem
rem  Environment: PYTHON, CUDA_VISIBLE_DEVICES, OUTDIR.
rem ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set "DO_TRAIN=0"
:parse
if not "%~1"=="" (
    if /i "%~1"=="--train"  set "DO_TRAIN=1"
    shift
    goto parse
)

if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] '%PYTHON%' has no PyTorch. Set PYTHON=C:\path\to\python.exe.
    exit /b 1
)

if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"
if not defined OUTDIR set "OUTDIR=reproduce_w12_v4_all"

set "CKPT=logs\20261111_215452_bridge_w12_v4_all\final_model.pth"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

if "%DO_TRAIN%"=="1" (
    echo === training bridge_w12_v4_all ^(400 epochs, no early stop^) on GPU %CUDA_VISIBLE_DEVICES% ===
    echo     the validation numbers printed below are in-sample; do not select on them
    "%PYTHON%" train_model.py --domain bridge_w12_v4_all
    if errorlevel 1 exit /b 1
    for /f "delims=" %%D in ('dir /b /ad /o-d logs\*_bridge_w12_v4_all') do (
        set "CKPT=logs\%%D\final_model.pth"
        goto trained
    )
    :trained
    echo === trained checkpoint: !CKPT! ===
)

if not exist "!CKPT!" (
    echo [ERROR] no such checkpoint: !CKPT!
    exit /b 1
)

for %%P in (single overlap_mirror) do (
    echo.
    echo === %%P ^(protocol %%P^) ===
    "%PYTHON%" evaluate_full.py --domain bridge_w12_v4 --protocol %%P ^
        --model_weights "!CKPT!" --out "%OUTDIR%\%%P.json" --overwrite
    if errorlevel 1 exit /b 1
)
echo.
echo === tls3_single ===
"%PYTHON%" evaluate_full.py --domain bridge_w12_v4 --protocol single --model_weights "!CKPT!" ^
    --out "%OUTDIR%\tls3_single.json" --overwrite --processed_data_path bridge/processed_tls3
if errorlevel 1 exit /b 1
echo.
echo === mls3_single ===
"%PYTHON%" evaluate_full.py --domain bridge_w12_v4 --protocol single --model_weights "!CKPT!" ^
    --out "%OUTDIR%\mls3_single.json" --overwrite --processed_data_path bridge/processed_mls
if errorlevel 1 exit /b 1

echo.
echo === summary ===
"%PYTHON%" -c "import json,os,sys;d=sys.argv[1];[print('%%-20s mIoU %%6.2f  OA %%6.2f  (%%s points)' %% (t, m['mIoU']*100, m['accuracy']*100, format(m['total_points'],','))) for t in ('single','overlap_mirror','tls3_single','mls3_single') if os.path.exists(os.path.join(d,t+'.json')) for m in [json.load(open(os.path.join(d,t+'.json')))['overall_metrics']]]" "%OUTDIR%"
endlocal
