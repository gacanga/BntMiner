@echo off
REM ===========================================================================
REM build_gpu_dll.bat - builds <dir>\bgpu.dll (CUDA Argon2id engine) with nvcc + MSVC.
REM   Target : NVIDIA P104-100 (Pascal, sm_61) + PTX, CUDA 12.9, Visual Studio 2022.
REM   Output : <dir>\bgpu.dll, bgpu.lib, cudart64_12.dll   (dir = %1, default dev)
REM   Env    : BM_NVCC_EXTRA = extra nvcc flags (build_all.bat reads run\nvcc_extra.txt)
REM   Log    : logs\build_gpu_dll.log
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0.."
set "OUT=%~1"
if "%OUT%"=="" set "OUT=dev"
if not exist logs mkdir logs
if not exist "%OUT%" mkdir "%OUT%"
set "LOG=%CD%\logs\build_gpu_dll.log"
(echo [%date% %time%] build_gpu_dll start> "%LOG%") 2>nul || set "LOG=%CD%\logs\build_gpu_dll_%RANDOM%.log"

set "CUDA129=%CUDA_PATH_V12_9%"
if "%CUDA129%"=="" set "CUDA129=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not exist "%CUDA129%\bin\nvcc.exe" (
    echo ERROR: CUDA 12.9 nvcc.exe not found >> "%LOG%"
    echo ERROR: CUDA 12.9 nvcc.exe not found.
    exit /b 1
)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
)
if "%VSINSTALL%"=="" set "VSINSTALL=C:\Program Files\Microsoft Visual Studio\2022\Community"
if not exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" (
    echo ERROR: vcvars64.bat not found under "%VSINSTALL%" >> "%LOG%"
    echo ERROR: MSVC x64 tools not found.
    exit /b 1
)
set VSCMD_SKIP_SENDTELEMETRY=1
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: MSVC environment setup failed >> "%LOG%"
    echo ERROR: MSVC environment setup failed.
    exit /b 1
)
set "PATH=%CUDA129%\bin;%PATH%"
nvcc --version >> "%LOG%" 2>&1

if exist "%OUT%\bgpu.dll" del /q "%OUT%\bgpu.dll"
if exist "%OUT%\bgpu.dll" (
    echo ERROR: %OUT%\bgpu.dll is in use and cannot be replaced >> "%LOG%"
    echo ERROR: %OUT%\bgpu.dll is in use
    exit /b 1
)
echo [%date% %time%] compiling gpu\bnt_gpu.cu >> "%LOG%"
echo   compiling bnt_gpu.cu ...
nvcc -O3 %BM_NVCC_EXTRA% -gencode arch=compute_61,code=[sm_61,compute_61] -Wno-deprecated-gpu-targets -allow-unsupported-compiler -std=c++17 -Igpu -Isrc ^
     -Xptxas -v -Xcompiler "/MD /O2 /EHsc /W3" -cudart shared --shared ^
     -o "%OUT%\bgpu.dll" gpu\bnt_gpu.cu >> "%LOG%" 2>&1
if not exist "%OUT%\bgpu.dll" (
    echo [%date% %time%] bnt_gpu.cu FAILED >> "%LOG%"
    echo ERROR: bgpu.dll build failed, see logs\build_gpu_dll.log
    exit /b 1
)
copy /y "%CUDA129%\bin\cudart64_12.dll" "%OUT%\" >> "%LOG%" 2>&1
dir /b "%OUT%\*.dll" >> "%LOG%" 2>&1
echo [%date% %time%] build_gpu_dll OK >> "%LOG%"
echo bgpu.dll OK
exit /b 0
