#!/usr/bin/env python
"""
Startup script for the Building Footprint API
"""

import uvicorn
from minimal_app import app

if __name__ == "__main__":
    print("🚀 Starting Building Footprint API with Complete ML Pipeline...")
    print("📊 Available endpoints:")
    print("   - GET  /api-summary          : Complete API capabilities")
    print("   - POST /extract-buildings    : Extract building footprints from images")
    print("   - POST /process-state-data   : Process specific state datasets")
    print("   - POST /batch-process        : Batch process multiple states")
    print("   - GET  /model-performance    : View model performance metrics")
    print("   - GET  /docs                 : Interactive API documentation")
    print("")
    print("🌐 API will be available at: http://127.0.0.1:8003")
    print("📖 Documentation at: http://127.0.0.1:8003/docs")
    print("")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8003, 
        reload=True,
        log_level="info"
    )