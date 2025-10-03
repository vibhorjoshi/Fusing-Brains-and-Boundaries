#!/usr/bin/env python3
"""
Local Deployment Script for Real USA Agricultural Detection System
This script helps deploy the system locally without Docker
"""

import os
import sys
import subprocess
import platform
import time
import signal
import atexit
import webbrowser
import threading

# Global variables for process management
processes = []

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")

def run_command(command, description=None, wait=True, shell=True):
    """Run a command and optionally wait for it to finish"""
    if description:
        print(f"🚀 {description}...")
    
    print(f"Running: {command}")
    
    try:
        if wait:
            # Run and wait for completion
            result = subprocess.run(
                command, 
                shell=shell, 
                check=False,
                text=True,
                capture_output=True
            )
            
            if result.stdout:
                print(result.stdout.strip())
            
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
                
            return result.returncode == 0
        else:
            # Run in background
            if platform.system() == "Windows":
                process = subprocess.Popen(
                    command,
                    shell=shell,
                    text=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                process = subprocess.Popen(
                    command,
                    shell=shell,
                    text=True
                )
            
            processes.append(process)
            print(f"Started process with PID: {process.pid}")
            return True
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print_header("Installing Dependencies")
    
    # Check if pip is available
    if not run_command(f"{sys.executable} -m pip --version", "Checking pip installation"):
        print("❌ pip not available. Please install pip first.")
        return False
        
    # Install required packages
    requirements = [
        "fastapi",
        "uvicorn",
        "streamlit",
        "python-dotenv",
        "numpy",
        "opencv-python",
        "pandas",
        "pillow"
    ]
    
    for package in requirements:
        run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
    
    # Install from requirements.txt if it exists
    if os.path.exists("requirements.txt"):
        run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing from requirements.txt")
        
    return True

def create_env_file(env):
    """Create environment file if it doesn't exist"""
    env_file = f".env.{env}"
    if not os.path.exists(env_file):
        print(f"Creating environment file: {env_file}")
        with open(env_file, "w") as f:
            f.write(f"ENVIRONMENT={env}\n")
            f.write("API_HOST=localhost\n")
            f.write("API_PORT=8002\n")
            f.write("USE_GPU=false\n")
    else:
        print(f"Using existing environment file: {env_file}")
    
    # Set environment variables
    os.environ["ENVIRONMENT"] = env
    os.environ["API_HOST"] = "localhost"
    os.environ["API_PORT"] = "8002"
    
    return True

def start_backend_api():
    """Start the FastAPI backend server"""
    print_header("Starting Backend API")
    
    # Check if API file exists
    api_files = ["enhanced_adaptive_fusion_api.py", "api.py", "app.py", "main.py"]
    api_file = None
    
    for file in api_files:
        if os.path.exists(file):
            api_file = file
            break
    
    if not api_file:
        print("❌ No API file found. Please create an API file first.")
        return False
    
    # Start the API server in a new terminal window
    if platform.system() == "Windows":
        command = f"start cmd /k {sys.executable} -m uvicorn {api_file.replace('.py', '')}:app --host 0.0.0.0 --port 8002 --reload"
    else:
        command = f"gnome-terminal -- {sys.executable} -m uvicorn {api_file.replace('.py', '')}:app --host 0.0.0.0 --port 8002 --reload"
    
    return run_command(command, "Starting FastAPI backend server", wait=False, shell=True)

def start_streamlit_app():
    """Start the Streamlit app"""
    print_header("Starting Streamlit App")
    
    # Check if Streamlit app exists
    streamlit_files = ["streamlit_app.py", "app.py", "dashboard.py"]
    streamlit_file = None
    
    for file in streamlit_files:
        if os.path.exists(file):
            streamlit_file = file
            break
    
    if not streamlit_file:
        print("❌ No Streamlit app file found. Please create a Streamlit app file first.")
        return False
    
    # Start the Streamlit app in a new terminal window
    if platform.system() == "Windows":
        command = f"start cmd /k {sys.executable} -m streamlit run {streamlit_file} --server.port 8501"
    else:
        command = f"gnome-terminal -- {sys.executable} -m streamlit run {streamlit_file} --server.port 8501"
    
    return run_command(command, "Starting Streamlit app", wait=False, shell=True)

def open_dashboard():
    """Open the dashboard in a browser"""
    print_header("Opening Dashboard")
    
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(5)
    
    # Open dashboard in browser
    url = "http://localhost:8501"
    print(f"Opening dashboard at {url}")
    webbrowser.open(url)
    
    # Open API docs
    api_url = "http://localhost:8002/docs"
    print(f"API documentation available at {api_url}")
    
    return True

def cleanup_processes():
    """Clean up background processes on exit"""
    for process in processes:
        if process.poll() is None:  # If process is still running
            try:
                print(f"Terminating process {process.pid}")
                if platform.system() == "Windows":
                    subprocess.run(f"taskkill /F /PID {process.pid}", shell=True)
                else:
                    process.terminate()
            except Exception as e:
                print(f"Error terminating process: {str(e)}")

def main():
    """Main deployment function"""
    import argparse
    
    # Register cleanup function
    atexit.register(cleanup_processes)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nCtrl+C pressed. Cleaning up...")
        cleanup_processes()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Deploy GeoAI system locally")
    parser.add_argument("--env", "-e", choices=["development", "staging", "production"], 
                        default="development", help="Environment to deploy to")
    parser.add_argument("--skip-install", "-s", action="store_true",
                        help="Skip installing dependencies")
    parser.add_argument("--api-only", "-a", action="store_true",
                        help="Start only the API server")
    parser.add_argument("--streamlit-only", "-w", action="store_true",
                        help="Start only the Streamlit app")
    
    args = parser.parse_args()
    
    print_header(f"GeoAI Research System - Local Deployment ({args.env})")
    
    # Install dependencies
    if not args.skip_install and not install_dependencies():
        print("❌ Failed to install dependencies.")
        return
    
    # Create environment file
    create_env_file(args.env)
    
    # Start services based on flags
    if args.api_only:
        # Start only the API server
        if start_backend_api():
            print("✅ API server started.")
            print("📊 API Documentation: http://localhost:8002/docs")
    elif args.streamlit_only:
        # Start only the Streamlit app
        if start_streamlit_app():
            print("✅ Streamlit app started.")
            print("📊 Streamlit Dashboard: http://localhost:8501")
            open_dashboard()
    else:
        # Start both services
        if start_backend_api():
            print("✅ API server started.")
        
        if start_streamlit_app():
            print("✅ Streamlit app started.")
        
        open_dashboard()
    
    print_header("Deployment Complete")
    print("✅ System deployed locally!")
    print("📊 Dashboard URLs:")
    print("  - Streamlit Dashboard: http://localhost:8501")
    print("  - API Documentation: http://localhost:8002/docs")
    
    print("\n⚠️ Press Ctrl+C to stop all services when done.")
    print("   Or close the terminal windows directly.")
    
    # Keep the script running to manage the background processes
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down services...")

if __name__ == "__main__":
    main()