# Deploy and Test Dashboard
# This script helps quickly deploy and test the GeoAI Research Platform dashboard

# Exit on error
$ErrorActionPreference = "Stop"

# Display banner
Write-Host "===================================================================="
Write-Host "        GeoAI Research Platform - Dashboard Deployment Tool         "
Write-Host "===================================================================="
Write-Host ""

# Function to check if a command exists
function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $command) { return $true }
    }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Check Python installation
Write-Host "Checking Python installation..."
if (Test-CommandExists python) {
    $pythonVersion = (python --version 2>&1).ToString()
    Write-Host "✅ Python detected: $pythonVersion" -ForegroundColor Green
}
else {
    Write-Host "❌ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check if required files exist
Write-Host ""
Write-Host "Checking required files..."
$requiredFiles = @(
    "index.html",
    "alabama_cities_map.js",
    "live_pipeline_visualizer.js",
    "unified_backend.py"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ Found: $file" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host ""
    Write-Host "Some required files are missing. Do you want to continue anyway? (y/n)" -ForegroundColor Yellow
    $continue = Read-Host
    if ($continue -ne "y") {
        Write-Host "Deployment aborted." -ForegroundColor Red
        exit 1
    }
}

# Install required packages
Write-Host ""
Write-Host "Installing required Python packages..."

# Check if requirements.txt exists
if (Test-Path "requirements.txt") {
    Write-Host "Installing packages from requirements.txt..."
    python -m pip install -r requirements.txt
}
else {
    # Install minimum required packages
    Write-Host "requirements.txt not found. Installing minimum required packages..."
    python -m pip install fastapi uvicorn numpy
}

# Setup deployment options
Write-Host ""
Write-Host "Dashboard Deployment Configuration"
Write-Host "=================================="

# Ask for backend port
Write-Host "Backend port (default: 8002): " -NoNewline
$backendPort = Read-Host
if (-not $backendPort) { $backendPort = 8002 }

# Ask for frontend port
Write-Host "Frontend port (default: 3003): " -NoNewline
$frontendPort = Read-Host
if (-not $frontendPort) { $frontendPort = 3003 }

# Ask if should open in browser
Write-Host "Open in browser after deployment? (y/n, default: y): " -NoNewline
$openBrowser = Read-Host
$noBrowserFlag = ""
if ($openBrowser -eq "n") { $noBrowserFlag = "--no-browser" }

# Start deployment
Write-Host ""
Write-Host "Starting deployment..." -ForegroundColor Cyan
Write-Host "===================================================================="

# Deploy using the Python script if it exists, otherwise do manual deployment
if (Test-Path "deploy_dashboard.py") {
    python deploy_dashboard.py --backend-port $backendPort --frontend-port $frontendPort $noBrowserFlag
}
else {
    # Start backend server
    Write-Host "Starting backend server on port $backendPort..."
    $backendCmd = "python unified_backend.py --port $backendPort"
    Start-Process -WindowStyle Minimized powershell -ArgumentList "-Command", $backendCmd
    Write-Host "✅ Backend server started: http://localhost:$backendPort" -ForegroundColor Green
    
    # Wait a moment to ensure backend starts
    Start-Sleep -Seconds 2
    
    # Start frontend server
    Write-Host "Starting frontend server on port $frontendPort..."
    $frontendCmd = "python -m http.server $frontendPort"
    Start-Process -WindowStyle Minimized powershell -ArgumentList "-Command", $frontendCmd
    Write-Host "✅ Frontend server started: http://localhost:$frontendPort" -ForegroundColor Green
    
    # Open browser if requested
    if ($openBrowser -ne "n") {
        Start-Sleep -Seconds 2
        Write-Host "Opening dashboard in browser..."
        Start-Process "http://localhost:$frontendPort"
    }
    
    Write-Host ""
    Write-Host "Dashboard is now running!" -ForegroundColor Green
    Write-Host "Backend API: http://localhost:$backendPort"
    Write-Host "Frontend Dashboard: http://localhost:$frontendPort"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the servers."
    
    # Keep script running to prevent PowerShell window from closing
    while ($true) {
        Start-Sleep -Seconds 10
    }
}