@echo off
rem ============================================================
rem  Score a checkpoint on the geometry its domain trained it on.
rem
rem  --domain reads the block geometry, voxel lattice and
rem  architecture back out of domains\<name>.json. This is not a
rem  convenience: w6 and w12 were once scored at the default 2 m
rem  window and full resolution and came out 3.9 mIoU low, which
rem  produced three wrong conclusions before the mismatch was found.
rem
rem  Usage:
rem    run_domain_eval.bat <domain> <checkpoint.pth> [protocol] [out.json]
rem
rem  protocol (default overlap_mirror, the reported one):
rem    single          one view, stride = window    comparable to the paper baselines
rem    overlap         stride = window/2
rem    mirror          2 views (identity + mirrored)
rem    overlap_mirror  both                         +0.8 mIoU over single on w6
rem ============================================================
setlocal
cd /d "%~dp0"

if "%~2"=="" (
    echo usage: %~nx0 ^<domain^> ^<checkpoint.pth^> [protocol] [out.json]
    exit /b 1
)
set "DOMAIN=%~1"
set "CKPT=%~2"
set "PROTOCOL=%~3"
if "%PROTOCOL%"=="" set "PROTOCOL=overlap_mirror"
set "OUT=%~4"
if "%OUT%"=="" set "OUT=%~dp2score_%PROTOCOL%_%~n2.json"

if not exist "%CKPT%" (
    echo [ERROR] no such checkpoint: %CKPT%
    exit /b 1
)

if not defined PYTHON set "PYTHON=python"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"

echo === scoring %CKPT% as domain '%DOMAIN%', protocol '%PROTOCOL%' ===
"%PYTHON%" evaluate_full.py --domain %DOMAIN% --protocol %PROTOCOL% ^
    --model_weights "%CKPT%" --out "%OUT%"
endlocal
