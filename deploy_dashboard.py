#!/usr/bin/env python
"""
Dashboard Deployment Script
---------------------------
This script deploys the GeoAI Research Platform dashboard by:
1. Starting the backend API server
2. Setting up a simple HTTP server for the frontend
3. Opening the dashboard in a web browser
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse
import logging
import socket
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_BACKEND_PORT = 8002
DEFAULT_FRONTEND_PORT = 3003
DEFAULT_HOST = "localhost"

def is_port_in_use(port, host=DEFAULT_HOST):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True

def find_available_port(start_port, host=DEFAULT_HOST):
    """Find an available port starting from start_port."""
    port = start_port
    while is_port_in_use(port, host):
        port += 1
        if port > start_port + 100:  # Avoid infinite loop
            raise Exception(f"Could not find an available port after {start_port + 100}")
    return port

def run_backend_server(port):
    """Run the backend API server."""
    try:
        # Check if unified_backend.py exists
        if os.path.exists("unified_backend.py"):
            backend_script = "unified_backend.py"
        else:
            # Look for alternative backend scripts
            alternatives = ["app.py", "main.py", "server.py", "api.py"]
            backend_script = next((script for script in alternatives if os.path.exists(script)), None)
            
            if not backend_script:
                logger.error("Backend script not found. Please make sure unified_backend.py exists.")
                return False
        
        # Construct command to run backend server
        if sys.platform == "win32":
            command = f"start /B python {backend_script} --port {port}"
            subprocess.Popen(command, shell=True)
        else:
            command = f"python {backend_script} --port {port}"
            subprocess.Popen(command, shell=True)
            
        logger.info(f"Started backend server at http://{DEFAULT_HOST}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start backend server: {str(e)}")
        return False

def run_frontend_server(port):
    """Run a simple HTTP server for the frontend."""
    try:
        # Use Python's built-in HTTP server
        if sys.platform == "win32":
            command = f"start /B python -m http.server {port}"
            subprocess.Popen(command, shell=True)
        else:
            command = f"python -m http.server {port} &"
            subprocess.Popen(command, shell=True)
            
        logger.info(f"Started frontend server at http://{DEFAULT_HOST}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start frontend server: {str(e)}")
        return False

def wait_for_server(port, timeout=30):
    """Wait for server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((DEFAULT_HOST, port))
                return True
        except (socket.error, socket.timeout):
            time.sleep(0.5)
    return False

def open_dashboard(port):
    """Open the dashboard in a web browser."""
    url = f"http://{DEFAULT_HOST}:{port}"
    
    try:
        webbrowser.open(url)
        logger.info(f"Opened dashboard in web browser: {url}")
        return True
    except Exception as e:
        logger.error(f"Failed to open dashboard in web browser: {str(e)}")
        logger.info(f"Please manually open the dashboard at: {url}")
        return False

def main():
    """Main function to deploy the dashboard."""
    parser = argparse.ArgumentParser(description="Deploy the GeoAI Research Platform dashboard")
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT, help=f"Backend server port (default: {DEFAULT_BACKEND_PORT})")
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT, help=f"Frontend server port (default: {DEFAULT_FRONTEND_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the dashboard in a web browser")
    parser.add_argument("--backend-only", action="store_true", help="Start only the backend server")
    parser.add_argument("--frontend-only", action="store_true", help="Start only the frontend server")
    args = parser.parse_args()
    
    # Find available ports if the specified ports are in use
    backend_port = args.backend_port
    frontend_port = args.frontend_port
    
    if not args.frontend_only and is_port_in_use(backend_port):
        new_port = find_available_port(backend_port)
        logger.warning(f"Port {backend_port} is in use. Using port {new_port} for the backend server.")
        backend_port = new_port
    
    if not args.backend_only and is_port_in_use(frontend_port):
        new_port = find_available_port(frontend_port)
        logger.warning(f"Port {frontend_port} is in use. Using port {new_port} for the frontend server.")
        frontend_port = new_port
    
    # Start the backend server
    backend_started = False
    if not args.frontend_only:
        backend_started = run_backend_server(backend_port)
        if backend_started:
            if not wait_for_server(backend_port):
                logger.warning("Backend server might not be fully started yet.")
    
    # Start the frontend server
    frontend_started = False
    if not args.backend_only:
        frontend_started = run_frontend_server(frontend_port)
        if frontend_started:
            if not wait_for_server(frontend_port):
                logger.warning("Frontend server might not be fully started yet.")
    
    # Open the dashboard in a web browser
    if not args.no_browser and frontend_started:
        # Wait a moment for servers to initialize
        time.sleep(2)
        open_dashboard(frontend_port)
    
    # Display instructions
    logger.info("\nDashboard Deployment Summary:")
    if backend_started:
        logger.info(f"- Backend API: http://{DEFAULT_HOST}:{backend_port}")
    if frontend_started:
        logger.info(f"- Frontend Dashboard: http://{DEFAULT_HOST}:{frontend_port}")
    
    logger.info("\nPress Ctrl+C to stop the servers.")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping servers...")

if __name__ == "__main__":
    main()