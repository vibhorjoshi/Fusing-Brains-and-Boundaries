#!/usr/bin/env python3
"""
GeoAI Platform Development Server Runner
Starts both backend and frontend services in development mode
"""

import subprocess
import sys
import time
import threading
import os
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server"""
    print("🚀 Starting GeoAI Backend Server...")
    backend_dir = Path(__file__).parent
    python_exe = backend_dir / ".venv" / "Scripts" / "python.exe"
    
    if not python_exe.exists():
        python_exe = "python"
    
    try:
        subprocess.run([
            str(python_exe), 
            "simple_demo_server.py"
        ], cwd=backend_dir, check=True)
    except KeyboardInterrupt:
        print("🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Backend server error: {e}")

def run_frontend():
    """Run the Next.js frontend server"""
    print("🎨 Starting GeoAI Frontend Server...")
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Wait for backend to start
    time.sleep(5)
    
    try:
        subprocess.run([
            "npm", "run", "dev"
        ], cwd=frontend_dir, check=True)
    except KeyboardInterrupt:
        print("🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Frontend server error: {e}")

def main():
    """Main function to run both services"""
    print("=" * 60)
    print("🌍 GeoAI Platform Development Environment")
    print("=" * 60)
    print()
    print("Backend:  http://127.0.0.1:8002")
    print("Frontend: http://127.0.0.1:3000")
    print("Unified:  http://127.0.0.1:3000/dashboard/unified")
    print()
    print("Press Ctrl+C to stop all services")
    print("=" * 60)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    try:
        # Start frontend in main thread
        run_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down GeoAI Platform...")
        sys.exit(0)

if __name__ == "__main__":
    main()