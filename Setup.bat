@echo off
echo Installing required Python libraries...

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Pip is not installed. Please install Python and ensure pip is included in your PATH.
    exit /b
)

REM Install diffusers and torch
pip install diffusers[torch] torch

REM Install Pillow (PIL)
pip install pillow

echo Installation complete.
pause