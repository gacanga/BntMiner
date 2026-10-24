@echo off
REM BntMiner - mines Blocknet (BNT) with the settings of bntminer.conf (same folder).
REM Edit bntminer.conf first: replace YOUR_BLOCKNET_ADDRESS with your address.
REM Restarts the miner 10 s after any exit; close this window to stop.
setlocal EnableExtensions
title BntMiner - miner
cd /d "%~dp0"
findstr /c:"YOUR_BLOCKNET_ADDRESS" "%~dp0bntminer.conf" >nul
if not errorlevel 1 (
    echo.
    echo  Edit bntminer.conf: replace YOUR_BLOCKNET_ADDRESS with your payout address.
    echo  Modifiez bntminer.conf : remplacez YOUR_BLOCKNET_ADDRESS par votre adresse de paiement.
    echo.
    pause
    exit /b 1
)
set CUDA_DEVICE_ORDER=PCI_BUS_ID
:loop
"%~dp0bntminer.exe" --config "%~dp0bntminer.conf" --log "%~dp0bntminer.log"
echo BntMiner exited (code %ERRORLEVEL%), restarting in 10 s - close this window to stop.
timeout /t 10 /nobreak >nul
goto loop
