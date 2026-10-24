@echo off
REM stop_bnt.bat - stops BntMiner: the auto-restart loop(s) (bnt_pool.bat),
REM then any remaining bntminer.exe.   stop_bnt.bat nopause  (no key press)
REM The other miners (NullMiner, MdsMiner...) are not touched.
taskkill /fi "WINDOWTITLE eq BntMiner - miner*" /t /f >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | Where-Object { $_.CommandLine -match 'bnt_pool\.bat' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 2 /nobreak >nul
taskkill /im bntminer.exe /t /f >nul 2>&1
cd /d "%~dp0"
if not exist logs mkdir logs
echo [%date% %time%] miner stopped by stop_bnt.bat>> logs\pool.log
tasklist /fi "imagename eq bntminer.exe" 2>nul | find /i "bntminer.exe" >nul
if errorlevel 1 (echo Mineur arrete.) else (echo bntminer.exe tourne encore.)
if /i not "%~1"=="nopause" pause
