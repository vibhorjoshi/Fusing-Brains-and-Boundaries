@echo off

REM Create necessary directories
mkdir logs 2>NUL
mkdir data 2>NUL
mkdir temp 2>NUL
mkdir models 2>NUL

REM Copy example .env file if .env doesn't exist
if not exist .env (
    copy .env.example .env
    echo Created .env file from .env.example
    echo Please edit .env file with your settings
)

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
    echo Created virtual environment
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Setup completed successfully
echo Run 'venv\Scripts\activate.bat' to activate the virtual environment
echo Run 'uvicorn main:app --reload' to start the API server