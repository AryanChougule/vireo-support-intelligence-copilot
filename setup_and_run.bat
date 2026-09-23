@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Vireo Support Intelligence - Setup

cd /d "%~dp0"

echo.
echo ==============================================
echo   Vireo Support Intelligence - Clean Setup
echo ==============================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_CMD=python"
    ) else (
        echo ERROR: Python was not found.
        echo Install Python 3.10+ and make sure it is available as "python" or "py".
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :fail
) else (
    echo [1/5] Virtual environment already exists.
)

echo [2/5] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [3/5] Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist "data\tickets.csv" (
    echo.
    echo WARNING: data\tickets.csv was not found.
    echo Put the Vireo assessment files inside the data folder and run this file again.
    echo.
    pause
    exit /b 0
)

echo [4/5] Running deterministic analysis...
".venv\Scripts\python.exe" run_analysis.py --data-dir data
if errorlevel 1 goto :fail

echo [5/5] Running tests...
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto :fail

echo.
echo ==============================================
echo   Starting Streamlit
echo   Close this window to stop the application.
echo ==============================================
echo.

".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof

:fail
echo.
echo ==============================================
echo   SETUP FAILED
echo ==============================================
echo Check the error above.
pause
exit /b 1
