@echo off
rem ============================================================
rem  Fetch the SemanticBridge scans and convert them to the
rem  format the pipeline reads. ~8 GB download, ~14 GB after
rem  conversion; nothing here is redistributed by this repo.
rem
rem  Usage (from anywhere):  scripts\get_bridge_data.bat
rem  Re-running is safe: the download resumes and conversion
rem  skips scenes that are already present.
rem
rem  Needs curl and tar (both ship with Windows 10 1803+).
rem ============================================================
setlocal
cd /d "%~dp0.."

set "RAW=bridge\raw"
set "RECORD=https://zenodo.org/api/records/18738225/files"
if not exist "%RAW%" mkdir "%RAW%"
if not defined PYTHON set "PYTHON=python"

for %%F in (tls_dataset mls_data) do (
    if not exist "%RAW%\%%F.zip" (
        echo == downloading %%F.zip
        curl -L -C - --retry 5 --retry-delay 10 -o "%RAW%\%%F.zip" "%RECORD%/%%F.zip/content"
    ) else (
        echo == %%F.zip already present
    )
)

if not exist "%RAW%\tls" mkdir "%RAW%\tls"
if not exist "%RAW%\mls" mkdir "%RAW%\mls"
tar -xf "%RAW%\tls_dataset.zip" -C "%RAW%\tls"
tar -xf "%RAW%\mls_data.zip" -C "%RAW%\mls"

echo == converting TLS scans -^> bridge\processed
"%PYTHON%" convert_dataset.py --dataset semanticbridge ^
    --input_dir "%RAW%\tls" --output_dir bridge\processed

if exist "%RAW%\mls\val" (
    echo == converting MLS scans -^> bridge\processed_mls
    "%PYTHON%" convert_dataset.py --dataset semanticbridge ^
        --input_dir "%RAW%\mls\val" --output_dir bridge\processed_mls --split test
    if not exist bridge\processed_tls3\test mkdir bridge\processed_tls3\test
    for %%B in (13 17 19) do copy /y "bridge\processed\test\bridge_%%B_fr_rtc.pt" "bridge\processed_tls3\test\" >nul
)

echo.
echo done. Next:  run_domain_eval.bat bridge_w6 logs\20260929_150153_bridge_w6\final_model.pth single
endlocal
