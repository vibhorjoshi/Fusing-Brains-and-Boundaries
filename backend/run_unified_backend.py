"""
Startup script for the Unified Building Footprint AI Backend

This script handles:
1. Environment setup
2. Dependencies check
3. ML pipeline initialization  
4. Backend server startup

Usage:
    python run_unified_backend.py --mode development
    python run_unified_backend.py --mode production
    python run_unified_backend.py --demo
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn", 
        "numpy",
        "gdal",
        "rasterio",
        "geopandas",
        "shapely"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "gdal":
                # Try multiple ways to import GDAL
                try:
                    from osgeo import gdal
                    logger.info(f"✅ {package} - OK (via osgeo)")
                except ImportError:
                    import rasterio
                    # If rasterio works, GDAL is available
                    logger.info(f"✅ {package} - OK (via rasterio)")
            else:
                __import__(package)
                logger.info(f"✅ {package} - OK")
        except ImportError:
            if package == "gdal":
                logger.warning(f"⚠️ {package} - Not directly importable but may be available via rasterio")
            else:
                missing_packages.append(package)
                logger.error(f"❌ {package} - MISSING")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with:")
        logger.info("pip install fastapi uvicorn numpy rasterio geopandas shapely")
        logger.info("conda install gdal -c conda-forge")
        return False
    
    logger.info("✅ All dependencies satisfied")
    return True

def setup_environment(mode: str):
    """Setup environment variables"""
    os.environ["ENVIRONMENT"] = mode
    
    if mode == "development":
        os.environ["HOST"] = "127.0.0.1"
        os.environ["PORT"] = "8000"
        os.environ["DEBUG"] = "true"
    elif mode == "production":
        os.environ["HOST"] = "0.0.0.0"
        os.environ["PORT"] = "8000"
        os.environ["DEBUG"] = "false"
    
    logger.info(f"🔧 Environment configured for {mode} mode")

def check_ml_pipeline():
    """Check if ML pipeline components are available"""
    try:
        from src.config import Config
        logger.info("✅ Config module found")
        # Try to import pipeline without initializing it
        import importlib.util
        spec = importlib.util.spec_from_file_location("pipeline", "src/pipeline.py")
        if spec is not None:
            logger.info("✅ ML Pipeline module found")
            return True
        else:
            logger.warning("⚠️ ML Pipeline module not found")
            return False
    except Exception as e:
        logger.warning(f"⚠️ ML Pipeline not available: {e}")
        logger.info("Backend will run in API-only mode")
        return False

def run_demo():
    """Run a quick demo of the system"""
    logger.info("🎯 Running system demo...")
    
    try:
        # Import and run a simple test
        import requests
        import time
        
        # Start server in background
        logger.info("Starting backend server...")
        
        # Give it a moment to start
        time.sleep(2)
        
        # Test health endpoint
        try:
            response = requests.get("http://127.0.0.1:8000/health")
            if response.status_code == 200:
                logger.info("✅ Health check passed")
                logger.info(f"Response: {response.json()}")
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Could not connect to server: {e}")
        
        # Test capabilities
        try:
            response = requests.get("http://127.0.0.1:8000/api/v1/capabilities")
            if response.status_code == 200:
                logger.info("✅ Capabilities check passed")
                capabilities = response.json()
                logger.info(f"ML Processing: {capabilities['ml_processing']}")
                logger.info(f"Supported formats: {capabilities['supported_formats']}")
            else:
                logger.error(f"❌ Capabilities check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Capabilities test failed: {e}")
            
    except ImportError:
        logger.warning("Requests not available for demo, install with: pip install requests")

def start_server(mode: str, reload: bool = False):
    """Start the unified backend server"""
    logger.info(f"🚀 Starting Unified Backend Server ({mode} mode)")
    
    # Import and run the server
    try:
        import uvicorn
        from unified_backend import app
        
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "127.0.0.1")
        
        logger.info(f"Server starting on http://{host}:{port}")
        logger.info(f"API Documentation: http://{host}:{port}/docs")
        logger.info(f"Health Check: http://{host}:{port}/health")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload and mode == "development",
            workers=1 if mode == "development" else 4,
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Unified Building Footprint AI Backend")
    parser.add_argument("--mode", choices=["development", "production"], default="development",
                       help="Run in development or production mode")
    parser.add_argument("--demo", action="store_true", help="Run system demo")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    
    args = parser.parse_args()
    
    logger.info("🏗️ Building Footprint AI - Unified Backend Startup")
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        sys.exit(1)
    
    if args.check_deps:
        logger.info("✅ Dependency check completed successfully")
        sys.exit(0)
    
    # Setup environment
    setup_environment(args.mode)
    
    # Check ML pipeline
    ml_available = check_ml_pipeline()
    
    if args.demo:
        run_demo()
        return
    
    # Start server
    start_server(args.mode, args.reload)

if __name__ == "__main__":
    main()