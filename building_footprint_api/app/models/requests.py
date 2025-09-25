"""
Request and response models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

from app.models.schemas import BoundingBox

class AuthRequest(BaseModel):
    """Authentication request"""
    username: str
    password: str

class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime

class ErrorResponse(BaseModel):
    """Standard error response"""
    status_code: int
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchProcessingRequest(BaseModel):
    """Batch processing request for multiple areas"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    processing_requests: List[Dict[str, Any]]
    notification_email: Optional[str] = None

class BatchStatusResponse(BaseModel):
    """Batch processing status"""
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    progress: float  # 0.0 to 1.0
    
class MapBoundsRequest(BaseModel):
    """Request model for map bounds"""
    bounds: BoundingBox
    zoom_level: int = Field(default=18, ge=10, le=22)
    width: int = Field(default=1280, ge=256, le=2048)
    height: int = Field(default=1280, ge=256, le=2048)
    maptype: str = Field(default="satellite", pattern="^(satellite|roadmap|hybrid|terrain)$")

class TileRequest(BaseModel):
    """Request model for map tiles"""
    bounds: BoundingBox
    zoom_level: int = Field(default=18, ge=10, le=22)
    maptype: str = Field(default="satellite", pattern="^(satellite|roadmap|hybrid|terrain)$")