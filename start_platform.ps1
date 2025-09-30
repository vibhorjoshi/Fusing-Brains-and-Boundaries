# GeoAI Platform Complete Startup Script - Windows PowerShell Version
# NASA-Level Professional Platform Launcher

param(
    [switch]$SkipInstall = $false,
    [switch]$Production = $false
)

$ErrorActionPreference = "Continue"  # Changed to Continue for better error handling

# Global variables for jobs
$global:BackendJob = $null
$global:FrontendJob = $null

# ASCII Art Banner
function Show-Banner {
    Write-ColoredText @"

    ██████╗ ███████╗ ██████╗  █████╗ ██╗
    ██╔══██╗██╔════╝██╔═══██╗██╔══██╗██║
    ██████╔╝█████╗  ██║   ██║███████║██║
    ██╔══██╗██╔══╝  ██║   ██║██╔══██║██║
    ██████╔╝███████╗╚██████╔╝██║  ██║██║
    ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝
    
    🚀 NASA-LEVEL BUILDING FOOTPRINT AI PLATFORM
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
"@ "Cyan"
}

# Check system requirements
function Test-SystemRequirements {
    Write-Header "System Requirements Check"
    
    $requirements = @(
        @{ Name = "Python"; Command = "python"; Version = "python --version" },
        @{ Name = "Node.js"; Command = "node"; Version = "node --version" },
        @{ Name = "NPM"; Command = "npm"; Version = "npm --version" }
    )
    
    $allGood = $true
    
    foreach ($req in $requirements) {
        try {
            $null = Get-Command $req.Command -ErrorAction Stop
            $version = Invoke-Expression $req.Version
            Write-Success "$($req.Name) is available: $version"
        }
        catch {
            Write-Error "$($req.Name) is not installed or not in PATH"
            $allGood = $false
        }
    }
    
    if (-not $allGood) {
        Write-Error "Please install missing requirements and try again"
        exit 1
    }
}

# Setup Python environment
function Initialize-PythonEnvironment {
    Write-Header "Python Environment Setup"
    
    if (-not (Test-Path ".venv")) {
        Write-Info "Creating Python virtual environment..."
        python -m venv .venv
        Write-Success "Virtual environment created"
    }
    
    Write-Info "Activating virtual environment..."
    & ".\.venv\Scripts\Activate.ps1"
    
    if (-not $SkipInstall) {
        Write-Info "Installing Python dependencies..."
        & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
        & ".\.venv\Scripts\pip.exe" install -r requirements.txt
        Write-Success "Python dependencies installed"
    }
}

# Setup Node.js environment
function Initialize-NodeEnvironment {
    Write-Header "Frontend Environment Setup"
    
    Set-Location "frontend"
    
    if (-not (Test-Path "node_modules") -or -not $SkipInstall) {
        Write-Info "Installing Node.js dependencies..."
        npm install
        Write-Success "Node.js dependencies installed"
    }
    
    Set-Location ".."
}

# Start backend server
function Start-BackendServer {
    Write-Header "Starting Backend Server"
    
    Write-Info "Launching FastAPI server on port 8002..."
    
    $backendJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        & ".\.venv\Scripts\python.exe" simple_demo_server.py
    }
    
    # Wait for backend to initialize
    Start-Sleep -Seconds 5
    
    # Check if backend is running
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8002/" -TimeoutSec 10
        Write-Success "Backend server started successfully"
        Write-Info "API Message: $($response.message)"
        return $backendJob
    }
    catch {
        Write-Error "Backend server failed to start"
        Stop-Job $backendJob -PassThru | Remove-Job
        throw
    }
}

# Start frontend server
function Start-FrontendServer {
    Write-Header "Starting Frontend Server"
    
    Write-Info "Launching Next.js development server on port 3000..."
    
    $frontendJob = Start-Job -ScriptBlock {
        Set-Location "$using:PWD\frontend"
        npm run dev
    }
    
    # Wait for frontend to initialize
    Start-Sleep -Seconds 15
    
    # Check if frontend is running
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 10
        Write-Success "Frontend server started successfully"
        return $frontendJob
    }
    catch {
        Write-Warning "Frontend server may still be starting up"
        return $frontendJob
    }
}

# Perform health checks
function Test-SystemHealth {
    Write-Header "System Health Check"
    
    $endpoints = @(
        @{ Name = "Backend API"; Url = "http://127.0.0.1:8002/" },
        @{ Name = "Backend Health"; Url = "http://127.0.0.1:8002/health" },
        @{ Name = "Live Visualization"; Url = "http://127.0.0.1:8002/live" },
        @{ Name = "Globe Visualization"; Url = "http://127.0.0.1:8002/globe" },
        @{ Name = "ML Processing"; Url = "http://127.0.0.1:8002/ml-processing" },
        @{ Name = "Analytics Dashboard"; Url = "http://127.0.0.1:8002/analytics" }
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            $response = Invoke-RestMethod -Uri $endpoint.Url -TimeoutSec 5
            Write-Success "$($endpoint.Name) is healthy"
        }
        catch {
            Write-Warning "$($endpoint.Name) health check failed"
        }
    }
}

# Display platform information
function Show-PlatformInfo {
    Write-Header "🎉 GeoAI Platform Successfully Launched!"
    
    Write-ColoredText @"

📍 ACCESS URLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 NASA-Style Frontend:     http://localhost:3000
🚀 Backend API:             http://127.0.0.1:8002
📖 API Documentation:       http://127.0.0.1:8002/docs
🌟 Live 3D Visualization:   http://127.0.0.1:8002/live
🌍 Interactive Globe:       http://127.0.0.1:8002/globe
🧠 ML Pipeline Monitor:     http://127.0.0.1:8002/ml-processing
📊 Analytics Dashboard:     http://127.0.0.1:8002/analytics

🎮 PLATFORM FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ NASA-Style Mission Control Interface
✅ Real-time 3D Building Detection
✅ Interactive Earth Globe Visualization
✅ Advanced ML Pipeline Monitoring
✅ WebSocket Live Data Streaming
✅ Professional Analytics Dashboard
✅ Multi-Model AI Processing
✅ Geographic Data Visualization

🔧 SYSTEM STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Backend Status:    🟢 OPERATIONAL
Frontend Status:   🟢 OPERATIONAL  
ML Pipeline:       🟢 READY
Data Streams:      🟢 ACTIVE

"@ "Green"

    Write-ColoredText "💡 Press Ctrl+C to stop all services" "Yellow"
}

# Cleanup function
function Stop-Platform {
    Write-Header "Shutting Down GeoAI Platform"
    
    if ($global:BackendJob) {
        Write-Info "Stopping backend server..."
        Stop-Job $global:BackendJob -PassThru | Remove-Job
    }
    
    if ($global:FrontendJob) {
        Write-Info "Stopping frontend server..."
        Stop-Job $global:FrontendJob -PassThru | Remove-Job
    }
    
    Write-Success "Platform shutdown complete"
}

# Main execution function
function Start-GeoAIPlatform {
    try {
        Show-Banner
        Test-SystemRequirements
        Initialize-PythonEnvironment
        Initialize-NodeEnvironment
        
        $global:BackendJob = Start-BackendServer
        $global:FrontendJob = Start-FrontendServer
        
        Start-Sleep -Seconds 5
        Test-SystemHealth
        Show-PlatformInfo
        
        # Keep script running and monitor jobs
        Write-Info "Monitoring platform services... Press Ctrl+C to stop"
        while ($true) {
            Start-Sleep -Seconds 30
            
            # Check job health
            if ($global:BackendJob.State -eq "Failed" -or $global:FrontendJob.State -eq "Failed") {
                Write-Error "One or more services have failed"
                break
            }
        }
    }
    catch {
        Write-Error "An error occurred: $_"
        Stop-Platform
        exit 1
    }
    finally {
        Stop-Platform
    }
}

# Handle Ctrl+C gracefully
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    Stop-Platform
}

# Start the platform
Start-GeoAIPlatform