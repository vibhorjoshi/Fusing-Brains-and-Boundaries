"""
FastAPI Application Entry Point
Building Footprint Regularization API
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from app.core.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan - runs on startup and shutdown
    """
    # Startup logic
    logger.info("Starting Building Footprint API...")
    
    yield
    
    # Shutdown logic
    logger.info("Shutting down Building Footprint API...")

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }

# Test endpoint for debugging
@app.get("/debug")
async def debug():
    logger.info("Debug endpoint accessed")
    return {
        "debug": True,
        "settings": {
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )