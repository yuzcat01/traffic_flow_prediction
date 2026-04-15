@echo off
setlocal

cd /d "%~dp0"

REM Optional: set your conda env name here for double-click launch (e.g. traffic_gnn)
set "CONDA_ENV_NAME="

set "PYTHON_CMD="

if not defined PYTHON_CMD if defined CONDA_PREFIX set "PYTHON_CMD=python"
if not defined PYTHON_CMD if defined CONDA_EXE if not "%CONDA_ENV_NAME%"=="" set "PYTHON_CMD=%CONDA_EXE% run -n %CONDA_ENV_NAME% python"

if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
if not defined PYTHON_CMD if exist "venv\Scripts\python.exe" set "PYTHON_CMD=venv\Scripts\python.exe"

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERROR] Python not found.
    echo [TIP] If you use conda, activate env first then rerun:
    echo [TIP] conda activate ^<your_env^>
    pause
    exit /b 1
)

echo [INFO] Starting GUI...
%PYTHON_CMD% run_gui.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] GUI exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
