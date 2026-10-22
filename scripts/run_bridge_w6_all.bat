@echo off
rem ============================================================
rem  The headline SemanticBridge run: the 6 m / 4 cm recipe trained
rem  on every labelled block of the 15 training bridges.
rem
rem  Why this is a separate recipe from bridge_w6
rem  -------------------------------------------
rem  bridge_w6 holds out 20%% of the blocks for validation and early
rem  stopping. The published benchmark baselines were trained without
rem  a validation set, so that held-out fifth is a protocol difference
rem  and not a property of the model. bridge_w6_all removes it:
rem    --val_in_train         every labelled block goes into training
rem    num_epochs 400         fixed in advance, so nothing is selected
rem    early_stop_patience    effectively disabled
rem  The per-epoch validation numbers printed during training are
rem  therefore IN-SAMPLE. They must not be used to pick a checkpoint.
rem  Report final_model.pth. best_model.pth is not released for this
rem  run for exactly that reason.
rem
rem  Measured (84,153,822 points, official 15/5 split)
rem  -------------------------------------------------
rem     single          mIoU 70.04   OA 91.22   ^<- protocol matches the baselines
rem     overlap_mirror  mIoU 70.74   OA 91.82   ^<- headline; votes, baselines do not
rem  Repeating a pass moves the last digit: the single-view figure has been
rem  measured at both 70.04 and 70.05 on this checkpoint. Spread of about
rem  0.01 mIoU is normal here, so treat differences under 0.05 as noise.
rem  For reference, the same recipe with the validation split held out
rem  (bridge_w6) scores 69.81 single-view, and the two-scale fusion that
rem  gained 0.68 there does not help here: fusing this checkpoint with the
rem  12 m one gives 70.50, below voting alone.
rem
rem  Usage
rem  -----
rem    run_bridge_w6_all.bat              score the released checkpoint
rem    run_bridge_w6_all.bat --train      train from scratch first (~54 h on one card)
rem    run_bridge_w6_all.bat --fusion     also score the 12 m partner and fuse
rem
rem  Environment: PYTHON, CUDA_VISIBLE_DEVICES, OUTDIR.
rem ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set "DO_TRAIN=0"
set "DO_FUSION=0"
:parse
if not "%~1"=="" (
    if /i "%~1"=="--train"  set "DO_TRAIN=1"
    if /i "%~1"=="--fusion" set "DO_FUSION=1"
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
if not defined OUTDIR set "OUTDIR=reproduce_w6_all"

set "CKPT=logs\20261018_163423_bridge_w6_all\final_model.pth"
set "CKPT12=logs\20261001_050040_bridge_w12\final_model.pth"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

if "%DO_TRAIN%"=="1" (
    echo === training bridge_w6_all ^(400 epochs, no early stop^) on GPU %CUDA_VISIBLE_DEVICES% ===
    echo     the validation numbers printed below are in-sample; do not select on them
    "%PYTHON%" train_model.py --domain bridge_w6_all
    if errorlevel 1 exit /b 1
    rem train_model.py writes logs\<timestamp>_bridge_w6_all\; pick the newest one up.
    for /f "delims=" %%D in ('dir /b /ad /o-d logs\*_bridge_w6_all') do (
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

rem --domain bridge_w6: the scoring geometry is the recipe's, not the training variant's.
echo.
echo === single (protocol single) ===
"%PYTHON%" evaluate_full.py --domain bridge_w6 --protocol single ^
    --model_weights "!CKPT!" --out "%OUTDIR%\single.json" --overwrite
if errorlevel 1 exit /b 1

echo.
echo === overlap_mirror (protocol overlap_mirror) ===
"%PYTHON%" evaluate_full.py --domain bridge_w6 --protocol overlap_mirror ^
    --model_weights "!CKPT!" --out "%OUTDIR%\overlap_mirror.json" --overwrite ^
    --save_probs "%OUTDIR%\probs_w6all_om"
if errorlevel 1 exit /b 1

if "%DO_FUSION%"=="1" (
    if exist "%CKPT12%" (
        echo.
        echo === 12 m partner, same protocol, then equal-weight point-wise fusion ===
        "%PYTHON%" evaluate_full.py --domain bridge_w12 --protocol overlap_mirror ^
            --model_weights "%CKPT12%" --out "%OUTDIR%\w12_overlap_mirror.json" --overwrite ^
            --save_probs "%OUTDIR%\probs_w12_om"
        if errorlevel 1 exit /b 1
        "%PYTHON%" tools\fuse_votes.py --probs "%OUTDIR%\probs_w6all_om" "%OUTDIR%\probs_w12_om" ^
            --processed_data_path bridge/processed --test_area test ^
            --config configs/model_params_semanticbridge.json ^
            --out "%OUTDIR%\fusion_w6all_w12.json" --overwrite
        if errorlevel 1 exit /b 1
    ) else (
        echo [skip] no 12 m checkpoint at %CKPT12%; fusion not run
    )
)

echo.
echo === summary ===
"%PYTHON%" -c "import json,os,sys;d=sys.argv[1];[print('%%-20s mIoU %%6.2f  OA %%6.2f  (%%s points)' %% (t, m['mIoU']*100, m['accuracy']*100, format(m['total_points'],','))) for t in ('single','overlap_mirror','w12_overlap_mirror','fusion_w6all_w12') if os.path.exists(os.path.join(d,t+'.json')) for m in [json.load(open(os.path.join(d,t+'.json')))['overall_metrics']]]" "%OUTDIR%"
endlocal
