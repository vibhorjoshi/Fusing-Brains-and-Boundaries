"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import time
import logging
import psutil
import torch

from app.core.config import get_settings
from app.services.pipeline import PipelineService
from app.api.dependencies import get_pipeline_service

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check(
    pipeline_service: PipelineService = Depends(get_pipeline_service)
):
    """
    Detailed health check with system information
    """
    # Get system info
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Check model status
    model_loaded = pipeline_service.mask_rcnn is not None
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_used_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024 ** 3), 2),
            "disk_used_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        },
        "models": {
            "mask_rcnn_loaded": model_loaded,
            "model_path": settings.MODEL_PATH
        }
    }