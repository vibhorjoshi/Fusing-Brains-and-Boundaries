#!/usr/bin/env pwsh
# GeoAI Monitoring System Launcher (PowerShell)

Write-Host "Starting GeoAI Monitoring System..." -ForegroundColor Cyan
Write-Host ""

# Activate the Python environment if it exists
if (Test-Path ".venv_new\Scripts\Activate.ps1") {
    & .venv_new\Scripts\Activate.ps1
    Write-Host "Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "Warning: Virtual environment not found at .venv_new" -ForegroundColor Yellow
    Write-Host "Using system Python installation."
}

# Install required dependencies if needed
Write-Host "Installing required dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt
pip install psutil streamlit pandas matplotlib seaborn numpy

# Run the monitoring system
Write-Host ""
Write-Host "Starting monitoring system..." -ForegroundColor Green
python run_with_monitoring_fixed.py

# No need to deactivate in PowerShell as the script ends

Write-Host ""
Write-Host "Monitoring system stopped." -ForegroundColor Yellow
Read-Host -Prompt "Press Enter to exit"