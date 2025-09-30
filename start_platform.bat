@echo off
echo Starting GeoAI Platform...
echo.

REM Check if PowerShell is available
powershell -Command "Write-Host 'PowerShell is available'" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PowerShell is not available
    echo Please install PowerShell or use the manual setup
    pause
    exit /b 1
)

REM Run the PowerShell startup script
powershell -ExecutionPolicy Bypass -File "start_platform.ps1"

pause