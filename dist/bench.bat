@echo off
REM BntMiner - 120 s H/s benchmark on the GPUs of bntminer.conf. Stop mining first.
setlocal EnableExtensions
title BntMiner - bench
cd /d "%~dp0"
set CUDA_DEVICE_ORDER=PCI_BUS_ID
"%~dp0bntminer.exe" --config "%~dp0bntminer.conf" --bench 120
pause
