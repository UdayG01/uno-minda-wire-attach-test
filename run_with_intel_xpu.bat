@echo off
title Intel XPU Push Detector
echo ================================================
echo Intel XPU Push Detection System
echo ================================================
echo Setting Intel XPU optimization environment...
set SYCL_CACHE_PERSISTENT=1
set INTEL_EXTENSION_FOR_PYTORCH_CACHE_PERSISTENT=1

echo.
echo Environment configured for Intel Arc Graphics
echo.

echo Activating conda environment...
call conda activate venv
if %ERRORLEVEL% EQU 0 (
    echo ✅ Conda environment 'venv' activated
) else (
    echo ❌ Failed to activate conda environment 'venv'
    echo Please ensure conda is installed and the 'venv' environment exists
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo Starting push detector system with Intel XPU acceleration...
echo.
python src2/push_detector11.py

echo.
echo Push detector has stopped.
echo Press any key to exit...
pause >nul
