#!/usr/bin/env python3
"""
Startup script for GeoAI Research Backend
Handles environment setup and server startup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Check if pip is available
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        
        # Install requirements
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✅ Dependencies installed successfully")
        else:
            print("⚠️  Warning: requirements.txt not found")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Please install dependencies manually:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Setup environment variables and directories"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = ["data", "logs", "uploads", "models"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Set environment variables if not exists
    env_vars = {
        "ENVIRONMENT": "development",
        "LOG_LEVEL": "INFO",
        "HOST": "0.0.0.0",
        "PORT": "8002"
    }
    
    for key, default_value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = default_value
            print(f"🔧 Set {key}={default_value}")
    
    print("✅ Environment setup complete")

def check_port_availability(port):
    """Check if port is available"""
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"⚠️  Warning: Port {port} is already in use")
            return False
        return True
        
    except Exception as e:
        print(f"❌ Error checking port {port}: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting GeoAI Research Backend Server...")
    print("=" * 60)
    
    # Get settings
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8002"))
    
    # Check port availability
    if not check_port_availability(port):
        print(f"Try stopping existing process or use different port:")
        print(f"export PORT=8003")
        return False
    
    try:
        # Import and run the app
        from app.main import app
        import uvicorn
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False

def main():
    """Main startup function"""
    print("🛰️  GeoAI Research Backend Startup")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    if "--skip-deps" not in sys.argv:
        if not install_dependencies():
            print("❌ Dependency installation failed")
            return False
    
    # Start server
    return start_server()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)