#!/usr/bin/env python3
"""
Launch Enhanced Dashboard - Real USA Agricultural Detection System

This script starts both the frontend HTML server and the Streamlit backend,
ensuring proper coordination between the components.

Usage:
    python launch_enhanced_dashboard.py
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

# Define constants
FRONTEND_PORT = 8081
STREAMLIT_PORT = 8501
API_PORT = 8007
REPO_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def ensure_virtual_environment():
    """Ensure we're running in the virtual environment with required packages"""
    venv_path = REPO_DIR / ".venv_new"
    if not venv_path.exists():
        venv_path = REPO_DIR / ".venv"
    
    if not venv_path.exists():
        print("⚠️ Virtual environment not found. Using system Python.")
        return
    
    # Check if we're already in a virtual environment
    if os.environ.get("VIRTUAL_ENV"):
        print(f"✅ Using virtual environment: {os.environ['VIRTUAL_ENV']}")
        return
    
    # Activate the virtual environment
    if os.name == "nt":  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        if activate_script.exists():
            print(f"🔄 Activating virtual environment: {venv_path}")
            os.system(f'"{activate_script}"')
    else:  # Linux/Mac
        activate_script = venv_path / "bin" / "activate"
        if activate_script.exists():
            print(f"🔄 Activating virtual environment: {venv_path}")
            os.system(f'source "{activate_script}"')

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit")
            return False
    
    try:
        from src.open_source_geo_ai import OpenSourceGeoAI
        print("✅ GeoAI library is available")
    except ImportError:
        print("⚠️ GeoAI library (open_source_geo_ai.py) not properly configured.")
        print("   Some functionality may be limited.")
    
    return True

def start_streamlit_backend():
    """Start Streamlit backend server"""
    print(f"🚀 Starting Streamlit backend on port {STREAMLIT_PORT}...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(REPO_DIR / "streamlit_backend.py"),
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
    ]
    
    streamlit_process = subprocess.Popen(
        streamlit_cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a bit to ensure Streamlit starts
    time.sleep(2)
    
    if streamlit_process.poll() is not None:
        print("❌ Failed to start Streamlit backend")
        error_output = streamlit_process.stderr.read()
        print(f"Error: {error_output}")
        return None
    
    print(f"✅ Streamlit backend running at http://localhost:{STREAMLIT_PORT}")
    return streamlit_process

def start_frontend_server():
    """Start the HTML frontend server"""
    print(f"🚀 Starting frontend server on port {FRONTEND_PORT}...")
    frontend_cmd = [sys.executable, str(REPO_DIR / "serve_frontend.py")]
    
    frontend_process = subprocess.Popen(
        frontend_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a bit to ensure frontend starts
    time.sleep(2)
    
    if frontend_process.poll() is not None:
        print("❌ Failed to start frontend server")
        error_output = frontend_process.stderr.read()
        print(f"Error: {error_output}")
        return None
    
    print(f"✅ Frontend server running at http://localhost:{FRONTEND_PORT}")
    return frontend_process

def print_dashboard_urls():
    """Print URLs for accessing the dashboard components"""
    print("\n" + "=" * 60)
    print("🚀 Real USA Agricultural Detection System - Dashboard")
    print("=" * 60)
    print(f"📊 Frontend Dashboard: http://localhost:{FRONTEND_PORT}")
    print(f"📈 Streamlit Analytics: http://localhost:{STREAMLIT_PORT}")
    print(f"🔌 API Endpoint:       http://localhost:{API_PORT}")
    print("=" * 60)
    print("Press Ctrl+C to stop all servers")
    print("=" * 60 + "\n")

def open_dashboard_in_browser():
    """Open the dashboard in the default web browser"""
    # Open frontend dashboard
    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
    
    # Also open Streamlit dashboard in a new tab
    time.sleep(1)
    webbrowser.open(f"http://localhost:{STREAMLIT_PORT}")

def main():
    """Main function to launch the enhanced dashboard"""
    print("\n" + "=" * 60)
    print("🚀 Launching Enhanced Dashboard - Real USA Agricultural Detection System")
    print("=" * 60 + "\n")
    
    # Ensure we're in the correct directory
    os.chdir(REPO_DIR)
    
    # Ensure virtual environment
    ensure_virtual_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Aborting due to missing dependencies")
        return
    
    try:
        # Start Streamlit backend
        streamlit_process = start_streamlit_backend()
        if not streamlit_process:
            return
        
        # Start frontend server
        frontend_process = start_frontend_server()
        if not frontend_process:
            streamlit_process.terminate()
            return
        
        # Print URLs
        print_dashboard_urls()
        
        # Open dashboard in browser
        threading.Timer(2.0, open_dashboard_in_browser).start()
        
        # Keep the script running to maintain the processes
        try:
            # Monitor the processes and keep the main thread alive
            while True:
                if streamlit_process.poll() is not None:
                    print("⚠️ Streamlit process ended unexpectedly")
                    break
                
                if frontend_process.poll() is not None:
                    print("⚠️ Frontend server ended unexpectedly")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏹️ Shutting down servers...")
        finally:
            # Terminate processes when done
            if streamlit_process:
                streamlit_process.terminate()
            
            if frontend_process:
                frontend_process.terminate()
    
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
    
    print("\n✅ Dashboard servers shut down")

if __name__ == "__main__":
    main()
