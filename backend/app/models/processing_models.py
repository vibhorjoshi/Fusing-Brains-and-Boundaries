"""
Processing Models for GeoAI Research Backend
"""

from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    GEOTIFF = "geotiff"


class ModelType(str, Enum):
    MASK_RCNN = "mask_rcnn"
    UNET = "unet"
    DEEPLAB = "deeplab"
    ADAPTIVE_FUSION = "adaptive_fusion"


class SatelliteImage(BaseModel):
    """Satellite image data model"""
    id: Optional[int] = None
    filename: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    format: ImageFormat
    width: Optional[int] = None
    height: Optional[int] = None
    resolution: Optional[float] = None  # meters per pixel
    coordinates: Optional[Dict[str, float]] = None  # lat, lon bounds
    uploaded_at: Optional[datetime] = None
    uploaded_by: Optional[int] = None  # user_id
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingJob(BaseModel):
    """Processing job data model"""
    id: Optional[str] = None
    job_id: str = str(uuid.uuid4())
    user_id: int
    image_id: Optional[int] = None
    model_type: ModelType
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None  # seconds
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingSession(BaseModel):
    """Training session data model"""
    id: Optional[int] = None
    session_name: str
    region: str
    model_type: ModelType
    total_epochs: int
    current_epoch: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    metrics: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    dataset_info: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[int] = None  # user_id
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BuildingFootprint(BaseModel):
    """Building footprint detection result"""
    id: Optional[int] = None
    job_id: str
    building_id: str
    coordinates: List[List[float]]  # polygon coordinates
    confidence: float
    area: Optional[float] = None
    perimeter: Optional[float] = None
    shape_complexity: Optional[float] = None
    detected_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingRequest(BaseModel):
    """Processing request model"""
    image_url: Optional[HttpUrl] = None
    image_file: Optional[str] = None  # base64 encoded
    model_type: ModelType = ModelType.ADAPTIVE_FUSION
    parameters: Optional[Dict[str, Any]] = None
    apply_regularization: bool = True
    normalize: bool = True


class AdaptiveFusionRequest(BaseModel):
    """Adaptive fusion processing request"""
    mode: str = "live_adaptive"
    normalize: bool = True
    fusion_algorithm: str = "deep_attention"
    real_time: bool = False


class VectorConversionRequest(BaseModel):
    """Vector conversion request"""
    image_data: Optional[str] = None  # base64 encoded
    simplify_tolerance: float = 0.5
    smoothing_factor: float = 0.3
    corner_detection: bool = True
    geometric_normalization: bool = True
    output_format: str = "geojson"
    precision: str = "high"


class TrainingRequest(BaseModel):
    """Training request model"""
    region: str = "alabama"
    epochs: int = 50
    model_type: ModelType = ModelType.ADAPTIVE_FUSION
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.001
    dataset_path: Optional[str] = None