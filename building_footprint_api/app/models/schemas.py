"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime

class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_FOUND = "not_found"

class ProcessingStage(str, Enum):
    MASK_RCNN = "mask_rcnn"
    RT_REGULARIZATION = "rt_regularization"
    RR_REGULARIZATION = "rr_regularization"
    FER_REGULARIZATION = "fer_regularization"
    RL_FUSION = "rl_fusion"
    VECTORIZATION = "vectorization"

class BoundingBox(BaseModel):
    north: float
    south: float
    east: float
    west: float
    
    @validator('north', 'south')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('east', 'west')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

class ProcessingRequest(BaseModel):
    job_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    bounds: BoundingBox
    zoom_level: int = Field(default=18, ge=10, le=20)
    enable_rt: bool = True
    enable_rr: bool = True
    enable_fer: bool = True
    enable_rl_fusion: bool = True
    callback_url: Optional[str] = None

class BuildingFootprint(BaseModel):
    id: str
    geometry: Dict[str, Any]  # GeoJSON geometry
    properties: Dict[str, Any]
    confidence: float
    area: float

class ProcessingResult(BaseModel):
    job_id: str
    status: JobStatus
    buildings: List[BuildingFootprint]
    processing_time: float
    metrics: Dict[str, float]
    error_message: Optional[str] = None

class JobProgress(BaseModel):
    job_id: str
    stage: ProcessingStage
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ProcessingResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    result_url: Optional[str] = None
    
class MapResponse(BaseModel):
    image_base64: str
    width: int
    height: int
    bounds: Dict[str, float]
    zoom_level: int
    
class TileResponse(BaseModel):
    x: int
    y: int
    z: int
    url: str
    bounds: Dict[str, float]