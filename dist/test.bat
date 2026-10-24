@echo off
REM BntMiner - correctness tests: CPU reference, then every GPU against the reference
REM hashes (Rust argon2 crate). Stop mining first (the last test needs 3 x 2 GiB of GPU memory).
setlocal EnableExtensions
title BntMiner - test
cd /d "%~dp0"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
"%~dp0bntminer.exe" --config "%~dp0bntminer.conf" --test
echo.
echo Test exit code %ERRORLEVEL% (0 = all OK).
pause
