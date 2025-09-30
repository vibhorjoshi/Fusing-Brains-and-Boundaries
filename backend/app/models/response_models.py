"""
Response Models for GeoAI Research Backend
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from .processing_models import ProcessingStatus


class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingResult(BaseModel):
    """Processing result model"""
    job_id: str
    status: ProcessingStatus
    progress: float
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: datetime = datetime.now()
    version: str = "1.0.0"
    uptime: Optional[float] = None
    system_info: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingStatusResponse(BaseModel):
    """Training status response model"""
    session_id: Optional[int] = None
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 50
    progress: float = 0.0
    current_loss: Optional[float] = None
    current_accuracy: Optional[float] = None
    current_iou: Optional[float] = None
    estimated_time_remaining: Optional[float] = None
    metrics_history: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VisualizationResponse(BaseModel):
    """Visualization data response model"""
    timestamp: datetime = datetime.now()
    type: str
    status: str = "success"
    data: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStatsResponse(BaseModel):
    """System statistics response model"""
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    uptime: Optional[float] = None
    timestamp: datetime = datetime.now()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKeyValidationResponse(BaseModel):
    """API key validation response"""
    valid: bool
    component: Optional[str] = None
    access_level: Optional[str] = None
    expires_at: Optional[datetime] = None
    usage_count: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchProcessingResponse(BaseModel):
    """Batch processing response model"""
    batch_id: str
    total_jobs: int
    queued_jobs: int
    processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    estimated_completion: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }