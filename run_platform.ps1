# GeoAI Platform Development Runner - PowerShell Version
# Runs both backend and frontend servers concurrently

param(
    [switch]$Production = $false,
    [switch]$Docker = $false
)

$ErrorActionPreference = "Stop"

function Write-Banner {
    Write-Host "=" -ForegroundColor Cyan -NoNewline
    1..60 | ForEach-Object { Write-Host "=" -ForegroundColor Cyan -NoNewline }
    Write-Host ""
    Write-Host "🌍 GeoAI Platform Development Environment" -ForegroundColor Green
    Write-Host "=" -ForegroundColor Cyan -NoNewline
    1..60 | ForEach-Object { Write-Host "=" -ForegroundColor Cyan -NoNewline }
    Write-Host ""
    Write-Host ""
    Write-Host "Backend:  http://127.0.0.1:8002" -ForegroundColor Yellow
    Write-Host "Frontend: http://127.0.0.1:3000" -ForegroundColor Yellow
    Write-Host "Unified:  http://127.0.0.1:3000/dashboard/unified" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Red
    Write-Host "=" -ForegroundColor Cyan -NoNewline
    1..60 | ForEach-Object { Write-Host "=" -ForegroundColor Cyan -NoNewline }
    Write-Host ""
}

function Start-Backend {
    Write-Host "🚀 Starting GeoAI Backend Server..." -ForegroundColor Green
    
    $pythonExe = ".\.venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        $pythonExe = "python"
    }
    
    try {
        & $pythonExe simple_demo_server.py
    }
    catch {
        Write-Host "❌ Backend server error: $_" -ForegroundColor Red
    }
}

function Start-Frontend {
    Write-Host "🎨 Starting GeoAI Frontend Server..." -ForegroundColor Green
    
    # Wait for backend to start
    Start-Sleep -Seconds 3
    
    try {
        Set-Location "frontend"
        npm run dev
    }
    catch {
        Write-Host "❌ Frontend server error: $_" -ForegroundColor Red
    }
    finally {
        Set-Location ".."
    }
}

function Start-Docker {
    Write-Host "🐳 Starting GeoAI Platform with Docker..." -ForegroundColor Blue
    
    try {
        if ($Production) {
            docker-compose -f docker-compose.production.yml up --build
        } else {
            docker-compose up --build
        }
    }
    catch {
        Write-Host "❌ Docker error: $_" -ForegroundColor Red
    }
}

function Main {
    Write-Banner
    
    if ($Docker) {
        Start-Docker
        return
    }
    
    # Start backend in background job
    $backendJob = Start-Job -ScriptBlock ${function:Start-Backend}
    
    # Wait a moment for backend to initialize
    Start-Sleep -Seconds 2
    
    try {
        # Start frontend in foreground
        Start-Frontend
    }
    finally {
        # Clean up background job
        if ($backendJob) {
            Write-Host "🛑 Stopping backend server..." -ForegroundColor Yellow
            Stop-Job $backendJob
            Remove-Job $backendJob
        }
    }
}

# Trap Ctrl+C
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    Write-Host "`n🛑 Shutting down GeoAI Platform..." -ForegroundColor Red
}

# Run main function
Main