"""
Simple startup script for Building Footprint API
"""
import uvicorn
from app.core.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print(f"Starting Building Footprint API on {settings.HOST}:{settings.PORT}...")
    uvicorn.run("app.main:app", 
                host=settings.HOST, 
                port=settings.PORT, 
                reload=settings.DEBUG)
    print("Server is running. Press CTRL+C to quit.")