"""
Simple FastAPI server for GeoAI building footprint detection demo
Provides basic API endpoints without complex ML dependencies
"""

import asyncio
import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis imports with fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory storage")

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Initialize FastAPI app
app = FastAPI(
    title="GeoAI Building Footprint API",
    description="Advanced ML-powered building footprint detection and processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo purposes
jobs_storage: Dict[str, Dict] = {}
buildings_storage: Dict[str, Dict] = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        if self.active_connections:
            message_str = json.dumps(message)
            disconnected = set()
            
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")
                    disconnected.add(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.active_connections.discard(connection)

manager = ConnectionManager()

# Pydantic models
class BuildingExtractionRequest(BaseModel):
    image_url: str
    model_type: str = "mask_rcnn"  # mask_rcnn, adaptive_fusion, hybrid
    apply_regularization: bool = True
    confidence_threshold: float = 0.7

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    current_stage: str
    created_at: str
    updated_at: str
    results: Optional[Dict] = None
    error: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str
    features: Dict[str, bool]

# Helper functions
def create_job(job_type: str, data: Dict) -> str:
    """Create a new processing job"""
    job_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    jobs_storage[job_id] = {
        "job_id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0,
        "current_stage": "Initializing",
        "created_at": timestamp,
        "updated_at": timestamp,
        "data": data,
        "results": None,
        "error": None
    }
    
    return job_id

def update_job_progress(job_id: str, progress: int, stage: str, results: Optional[Dict] = None):
    """Update job progress"""
    if job_id in jobs_storage:
        jobs_storage[job_id].update({
            "progress": progress,
            "current_stage": stage,
            "updated_at": datetime.utcnow().isoformat(),
            "results": results
        })

async def simulate_ml_processing(job_id: str, model_type: str, image_url: str):
    """Simulate ML model processing with realistic stages and live updates"""
    try:
        stages = [
            (10, "Loading image data"),
            (25, f"Initializing {model_type} model"),
            (40, "Preprocessing image"),
            (60, "Running ML inference"),
            (80, "Applying regularization"),
            (95, "Post-processing results"),
            (100, "Complete")
        ]
        
        for progress, stage in stages:
            await asyncio.sleep(0.5)  # Simulate processing time
            update_job_progress(job_id, progress, stage)
            
            # Broadcast live update
            await manager.broadcast({
                "type": "job_progress",
                "job_id": job_id,
                "progress": progress,
                "stage": stage,
                "model_type": model_type
            })
        
        # Simulate realistic results
        buildings_detected = 15 + (hash(image_url) % 20)
        results = {
            "buildings_detected": buildings_detected,
            "processing_time": 2.3 + (hash(model_type) % 30) / 10,
            "model_used": model_type,
            "regularization_applied": True,
            "confidence_scores": [0.85, 0.92, 0.78, 0.89, 0.94],
            "image_dimensions": [1024, 1024, 3],
            "footprint_polygons": [],
            "building_locations": [
                {
                    "x": (hash(f"{job_id}_{i}") % 200) - 100,
                    "z": (hash(f"{job_id}_{i}_z") % 200) - 100,
                    "height": 5 + (hash(f"{job_id}_{i}_h") % 20),
                    "confidence": 0.7 + (hash(f"{job_id}_{i}_c") % 30) / 100
                }
                for i in range(buildings_detected)
            ]
        }
        
        jobs_storage[job_id]["status"] = "completed"
        jobs_storage[job_id]["results"] = results
        
        # Broadcast completion with building data
        await manager.broadcast({
            "type": "job_completed",
            "job_id": job_id,
            "results": results,
            "model_type": model_type
        })
        
        logger.info(f"Job {job_id} completed successfully with {model_type}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        jobs_storage[job_id]["status"] = "failed"
        jobs_storage[job_id]["error"] = str(e)
        
        # Broadcast error
        await manager.broadcast({
            "type": "job_failed",
            "job_id": job_id,
            "error": str(e)
        })

# API Routes

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "message": "GeoAI Building Footprint Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        features={
            "ml_processing": True,
            "building_detection": True,
            "regularization": True,
            "multi_model": True,
            "redis_caching": False,  # In-memory for demo
            "gpu_acceleration": False,  # Simulated
            "real_time_processing": True
        }
    )

@app.post("/api/v1/ml-processing/extract-buildings", response_model=JobResponse)
async def extract_buildings(
    request: BuildingExtractionRequest, 
    background_tasks: BackgroundTasks
):
    """Start building footprint extraction job"""
    try:
        # Validate model type
        valid_models = ["mask_rcnn", "adaptive_fusion", "hybrid"]
        if request.model_type not in valid_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model type. Must be one of: {valid_models}"
            )
        
        # Create job
        job_id = create_job("building_extraction", request.dict())
        
        # Start background processing
        background_tasks.add_task(
            simulate_ml_processing, 
            job_id, 
            request.model_type, 
            request.image_url
        )
        
        logger.info(f"Started building extraction job {job_id} with model {request.model_type}")
        
        return JobResponse(
            job_id=job_id,
            status="started",
            message=f"Building extraction started with {request.model_type} model"
        )
        
    except Exception as e:
        logger.error(f"Failed to start building extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status and results"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**jobs_storage[job_id])

@app.get("/api/v1/jobs")
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """List all jobs with optional filtering"""
    jobs = list(jobs_storage.values())
    
    if status:
        jobs = [job for job in jobs if job["status"] == status]
    
    jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)[:limit]
    
    return {
        "jobs": jobs,
        "total": len(jobs_storage),
        "filtered": len(jobs)
    }

@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs_storage[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/api/v1/models")
async def list_models():
    """List available ML models"""
    return {
        "models": [
            {
                "name": "mask_rcnn",
                "description": "Mask R-CNN for instance segmentation",
                "accuracy": 0.89,
                "speed": "medium",
                "gpu_required": False
            },
            {
                "name": "adaptive_fusion", 
                "description": "Adaptive feature fusion network",
                "accuracy": 0.92,
                "speed": "fast",
                "gpu_required": False
            },
            {
                "name": "hybrid",
                "description": "Hybrid CNN-Transformer architecture",
                "accuracy": 0.94,
                "speed": "slow",
                "gpu_required": True
            }
        ]
    }

@app.get("/api/v1/statistics")
async def get_statistics():
    """Get system statistics"""
    total_jobs = len(jobs_storage)
    completed_jobs = len([j for j in jobs_storage.values() if j["status"] == "completed"])
    failed_jobs = len([j for j in jobs_storage.values() if j["status"] == "failed"])
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
        "uptime": "Demo mode",
        "version": "1.0.0",
        "active_connections": len(manager.active_connections)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "message": "Connected to GeoAI Live Stream",
                "timestamp": datetime.utcnow().isoformat()
            }), 
            websocket
        )
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (ping/pong, etc.)
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}),
                        websocket
                    )
                elif data.get("type") == "subscribe":
                    # Client subscribing to specific updates
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "subscribed", 
                            "channels": data.get("channels", [])
                        }),
                        websocket
                    )
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": str(e)}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting GeoAI Building Footprint Detection API...")
    uvicorn.run(
        "simple_demo_server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )