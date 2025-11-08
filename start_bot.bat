@echo off
REM Forex RL Bot Environment Setup
REM This script ensures you're using the right Python environment

echo.
echo ========================================
echo FOREX RL TRADING BOT - ENVIRONMENT SETUP
echo ========================================
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo Found virtual environment. Activating...
    call .venv\Scripts\activate.bat
    echo Virtual environment activated!
) else (
    echo No virtual environment found. Using system Python.
    echo Make sure tensorboard is installed: python -m pip install tensorboard
)

echo.
echo You can now run:
echo   python quick_test.py           - Quick system verification
echo   python main.py --mode train    - Start training
echo   python demo_analytics.py      - View trade analytics
echo   python system_status.py       - Check system status
echo.
echo To exit this environment, type: deactivate
echo.

REM Keep the command prompt open
cmd /k
