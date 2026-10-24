@echo off
REM run_test.bat - correctness tests of dev\bntminer.exe: CPU reference against the Rust
REM argon2 vectors (m = 8 .. 65536 KiB), then on every selected GPU: the same vectors with
REM 4 kernel chunk sizes, and the real 2 GiB Blocknet PoW (3 vectors, one lane each).
REM Log: logs\test.log.  Extra arguments (e.g. --gpus 4) in run\test_args.txt
setlocal EnableExtensions
cd /d "%~dp0"
if not exist logs mkdir logs
set "PATH=%CD%\dev;C:\msys64\mingw64\bin;%PATH%"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set "ARGS="
if exist run\test_args.txt set "ARGS=--args-file run\test_args.txt"
echo [%date% %time%] test start> logs\test.log
if exist run\test_args.txt type run\test_args.txt>> logs\test.log
dev\bntminer.exe --test %ARGS% >> logs\test.log 2>&1
echo [%date% %time%] test exit=%ERRORLEVEL%>> logs\test.log
