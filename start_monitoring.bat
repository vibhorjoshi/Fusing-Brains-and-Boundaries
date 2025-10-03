@echo off
REM GeoAI Monitoring System Launcher
echo Starting GeoAI Monitoring System...
echo.

REM Activate the Python environment if it exists
if exist ".venv_new\Scripts\activate.bat" (
    call .venv_new\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo Warning: Virtual environment not found at .venv_new
    echo Using system Python installation.
)

REM Install required dependencies if needed
echo Installing required dependencies...
pip install -r requirements.txt
pip install psutil streamlit pandas matplotlib seaborn numpy

REM Run the monitoring system
echo.
echo Starting monitoring system...
python run_with_monitoring_fixed.py

REM Deactivate the environment on exit
if exist ".venv_new\Scripts\activate.bat" (
    call .venv_new\Scripts\deactivate.bat
)

echo.
echo Monitoring system stopped.
pause