@echo off
setlocal enabledelayedexpansion

REM Usage:
REM   build_exe.bat                     -> onedir + lite (default)
REM   build_exe.bat onedir              -> onedir + lite
REM   build_exe.bat onefile             -> onefile + lite
REM   build_exe.bat onedir lite         -> onedir + lite  (GUI only)
REM   build_exe.bat onefile full        -> onefile + full (include torch)
REM   build_exe.bat onedir lite nopause -> no pause at end (for CI / terminal)

set MODE=%1
if "%MODE%"=="" set MODE=onedir
set PROFILE=%2
if "%PROFILE%"=="" set PROFILE=lite
set PAUSE_ON_END=1
if /I "%3"=="nopause" set PAUSE_ON_END=0

set ENTRY=run_gui.py
set APP_NAME=TrafficFlowGUI
set ICON=src\gui\assets\app_icon.ico
set ADD_DATA=src/gui/assets;src/gui/assets
set BUILD_TAG=%RANDOM%%RANDOM%
set WORK_PATH=build\_pyi_work_%BUILD_TAG%

set COMMON_ARGS=--noconfirm -w --paths "src" --workpath "%WORK_PATH%" --name "%APP_NAME%" --icon "%ICON%" --add-data "%ADD_DATA%"
set EXTRA_ARGS=

echo.
python -c "import sys; print('[INFO] Python executable:', sys.executable)"
if errorlevel 1 (
    echo [ERROR] python command not found.
    if "%PAUSE_ON_END%"=="1" pause
    exit /b 1
)

if /I "%PROFILE%"=="full" (
    python -c "import torch; print('[INFO] torch version:', torch.__version__)"
    if errorlevel 1 (
        echo.
        echo [ERROR] Current Python environment has no torch.
        echo [ERROR] Please activate your training conda env and install torch first.
        echo [TIP] Example: conda activate traffic_gnn
        echo [TIP] Then rebuild with: build_exe.bat %MODE% full
        echo [TIP] Or use lite build: build_exe.bat %MODE% lite
        if "%PAUSE_ON_END%"=="1" pause
        exit /b 1
    )
    set EXTRA_ARGS=!EXTRA_ARGS! --collect-all torch
)

echo.
echo [INFO] Build mode: %MODE%
echo [INFO] Build profile: %PROFILE%
echo [INFO] Work path: %WORK_PATH%
echo [INFO] Entry: %ENTRY%
echo.

if /I "%MODE%"=="onefile" (
    python -m PyInstaller -F %COMMON_ARGS% %EXTRA_ARGS% "%ENTRY%"
) else (
    python -m PyInstaller -D %COMMON_ARGS% %EXTRA_ARGS% "%ENTRY%"
)

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    if "%PAUSE_ON_END%"=="1" pause
    exit /b 1
)

echo.
if /I "%MODE%"=="onefile" (
    echo [DONE] Output: dist\%APP_NAME%.exe
) else (
    echo [DONE] Output folder: dist\%APP_NAME%\
    echo [TIP] Run: dist\%APP_NAME%\%APP_NAME%.exe
)
echo [NOTE] Existing exe cannot be fixed by installing new packages later.
echo [NOTE] After dependency changes, please rebuild exe.
if "%PAUSE_ON_END%"=="1" pause

endlocal
