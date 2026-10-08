@echo off
rem ============================================================
rem  Train one domain recipe end to end.
rem
rem  Every per-model setting -- block geometry, voxel lattice,
rem  architecture flags, schedule, loss options -- lives in
rem  domains\<name>.json, so this script carries none of them and
rem  two models never drift apart because a flag was retyped.
rem
rem  Usage:   run_domain_train.bat <domain> [extra flags...]
rem
rem  Bridge domains (SemanticBridge, 15/5 official split):
rem    bridge          2 m window, full resolution     baseline, mIoU 65.23
rem    bridge_w6       6 m window, 4 cm voxels         BEST, mIoU 70.64
rem    bridge_w12      12 m window, 8 cm voxels        69.03
rem    bridge_w24      24 m window, 16 cm voxels       67.11 (coarse)
rem    bridge_w6_gpos  w6 + global-position channels   68.41  REJECTED -2.06
rem    bridge_w6_sol   w6 + structure-oriented loss    70.14  REJECTED -0.50
rem  Indoor:
rem    room            S3DIS chunk recipe
rem ============================================================
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo usage: %~nx0 ^<domain^> [extra flags...]
    echo available:
    for %%F in (domains\*.json) do echo   %%~nF
    exit /b 1
)

set "DOMAIN=%~1"
shift
set "ARGS="
:collect
if not "%~1"=="" (
    set "ARGS=%ARGS% %1"
    shift
    goto collect
)

if not defined PYTHON set "PYTHON=python"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"

echo === training domain '%DOMAIN%' on GPU %CUDA_VISIBLE_DEVICES% ===
"%PYTHON%" train_model.py --domain %DOMAIN%%ARGS%
endlocal
