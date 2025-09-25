"""
FastAPI Application Entry Point
Building Footprint Regularization API
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.api.endpoints import processing, health, maps
from app.core.config import get_settings
from app.core.websocket import sio_app
from app.utils.websocket_manager import ConnectionManager
from app.core.websocket_manager import WebSocketManager
from app.services.workflow import WorkflowManager
from app.services.pipeline import PipelineService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# WebSocket connection manager
manager = ConnectionManager()

# Initialize websocket manager
websocket_manager = WebSocketManager()

# Initialize pipeline service
pipeline_service = PipelineService()

# Workflow manager
workflow_manager = WorkflowManager(pipeline_service=pipeline_service, websocket_manager=websocket_manager)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Building Footprint API...")
    await workflow_manager.initialize()
    yield
    # Shutdown
    logger.info("Shutting down Building Footprint API...")
    await workflow_manager.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Building Footprint Regularization API",
    description="Scalable API for building footprint extraction and regularization using Hybrid GeoAI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Socket.IO
app.mount("/socket.io", socketio.ASGIApp(sio_app, other_asgi_app=app))

# Include API routers
app.include_router(processing.router, prefix="/api/v1", tags=["processing"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(maps.router, prefix="/api/v1", tags=["maps"])

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"Message: {data}", client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/")
async def root():
    return {"message": "Building Footprint Regularization API", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )