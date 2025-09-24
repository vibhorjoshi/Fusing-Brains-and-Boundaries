#!/usr/bin/env python3
"""
Enhanced Streamlit App Launcher
Provides multiple startup options for the building footprint extraction demo
"""

import sys
import subprocess
import argparse
import threading
import time

def start_streamlit_app(port=8501, host="0.0.0.0"):
    """Start the Streamlit application"""
    cmd = [
        "streamlit", "run", "streamlit_app.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    print(f"🚀 Starting Streamlit app on http://{host}:{port}")
    subprocess.run(cmd)

def start_api_server(port=8000):
    """Start the FastAPI server for API endpoints"""
    try:
        import uvicorn
        from streamlit_app import StreamlitDemo
        
        demo = StreamlitDemo()
        print(f"🔗 Starting API server on http://0.0.0.0:{port}")
        print(f"📚 API documentation: http://0.0.0.0:{port}/docs")
        
        uvicorn.run(demo.api_app, host="0.0.0.0", port=port, log_level="info")
    except ImportError:
        print("❌ FastAPI/Uvicorn not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

def start_hybrid_mode(streamlit_port=8501, api_port=8000):
    """Start both Streamlit and API servers"""
    print("🚀 Starting hybrid mode with both Streamlit and API servers...")
    
    # Start API server in background thread
    api_thread = threading.Thread(target=start_api_server, args=(api_port,), daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(2)
    
    # Start Streamlit in main thread
    start_streamlit_app(streamlit_port)

def main():
    parser = argparse.ArgumentParser(description="Launch Building Footprint Extraction Demo")
    parser.add_argument(
        "--mode", 
        choices=["streamlit", "api", "hybrid"], 
        default="streamlit",
        help="Launch mode: streamlit (web app), api (REST API), or hybrid (both)"
    )
    parser.add_argument(
        "--streamlit-port", 
        type=int, 
        default=8501,
        help="Port for Streamlit app (default: 8501)"
    )
    parser.add_argument(
        "--api-port", 
        type=int, 
        default=8000,
        help="Port for API server (default: 8000)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host address (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    print("🏗️ GPU-Accelerated Building Footprint Extraction")
    print("=" * 50)
    
    if args.mode == "streamlit":
        start_streamlit_app(args.streamlit_port, args.host)
    elif args.mode == "api":
        start_api_server(args.api_port)
    elif args.mode == "hybrid":
        start_hybrid_mode(args.streamlit_port, args.api_port)

if __name__ == "__main__":
    main()