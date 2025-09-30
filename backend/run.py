"""
Main entry point for GeoAI Research Backend
Production-ready unified backend server
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.main import app
from app.config.settings import get_settings

def main():
    """Main function to start the server"""
    
    # Get settings
    settings = get_settings()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    print("🚀 Starting GeoAI Research Backend Server")
    print(f"📊 Environment: {settings.environment}")
    print(f"🔧 Debug Mode: {settings.debug}")
    print(f"🌐 Server: http://{settings.host}:{settings.port}")
    print(f"📖 API Docs: http://{settings.host}:{settings.port}/docs")
    print(f"🏥 Health Check: http://{settings.host}:{settings.port}/health")
    print("=" * 60)
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
        workers=1 if settings.reload else 4
    )

if __name__ == "__main__":
    main()