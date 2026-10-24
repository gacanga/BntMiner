@echo off
REM run_bench.bat - H/s benchmark of dev\bntminer.exe (120 s by default).
REM   run_bench.bat profile : Argon2i / Argon2d halves for 1..N lanes (logs\profile.log)
REM Extra arguments (e.g. --gpus 4 --lanes 2 --bench 300) in run\bench_args.txt
REM Log: logs\bench.log (overwritten), history appended to logs\bench_history.log
setlocal EnableExtensions
cd /d "%~dp0"
if not exist logs mkdir logs
set "PATH=%CD%\dev;C:\msys64\mingw64\bin;%PATH%"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set "ARGS="
if exist run\bench_args.txt set "ARGS=--args-file run\bench_args.txt"
if /i "%~1"=="profile" (
    echo [%date% %time%] profile start> logs\profile.log
    if exist run\bench_args.txt type run\bench_args.txt>> logs\profile.log
    dev\bntminer.exe --profile %ARGS% >> logs\profile.log 2>&1
    echo [%date% %time%] profile exit=%ERRORLEVEL%>> logs\profile.log
    type logs\profile.log >> logs\bench_history.log
    goto :eof
)
echo [%date% %time%] bench start> logs\bench.log
if exist run\bench_args.txt type run\bench_args.txt>> logs\bench.log
dev\bntminer.exe --bench 120 %ARGS% >> logs\bench.log 2>&1
echo [%date% %time%] bench exit=%ERRORLEVEL%>> logs\bench.log
type logs\bench.log >> logs\bench_history.log
