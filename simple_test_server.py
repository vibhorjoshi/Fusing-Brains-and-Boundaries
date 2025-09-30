"""
Simple test server to verify the unified backend works
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set environment
os.environ["ENVIRONMENT"] = "development"

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    
    app = FastAPI(title="Building Footprint AI - Simple Test")
    
    @app.get("/")
    def root():
        return {"message": "Unified Backend is working!", "status": "healthy"}
    
    @app.get("/health")
    def health():
        try:
            import numpy as np
            from osgeo import gdal
            return {
                "status": "healthy",
                "message": "All basic dependencies working",
                "numpy_version": np.__version__,
                "gdal_available": True
            }
        except ImportError as e:
            return {
                "status": "partial", 
                "message": f"Some dependencies missing: {e}",
                "basic_functionality": True
            }
    
    @app.get("/test/capabilities")
    def capabilities():
        caps = {
            "basic_api": True,
            "geospatial_processing": False,
            "ml_pipeline": False
        }
        
        try:
            import rasterio
            import geopandas
            caps["geospatial_processing"] = True
        except ImportError:
            pass
        
        try:
            from src.config import Config
            caps["ml_pipeline"] = True
        except ImportError:
            pass
        
        return {"capabilities": caps}
    
    if __name__ == "__main__":
        print("🚀 Starting Simple Test Server...")
        print("📍 Health Check: http://127.0.0.1:8001/health")
        print("🔧 Capabilities: http://127.0.0.1:8001/test/capabilities")
        
        print("🏢 Simple Building Footprint Test Server")
        print("=" * 40)
        print("🔗 URL: http://127.0.0.1:8005")
        print("📖 Docs: http://127.0.0.1:8005/docs")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8005,
            reload=False,
            workers=1
        )

except Exception as e:
    print(f"❌ Failed to start server: {e}")
    print("This might be due to SSL/DLL issues with the Python installation")
    print("Try running the building_footprint_api/minimal_app.py instead")