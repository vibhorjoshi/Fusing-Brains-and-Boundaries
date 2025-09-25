"""
ML processing schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

class LayerType(str, Enum):
    """Available layer types for processing"""
    AVG = "avg"
    MAX = "max" 
    MIN = "min"
    SUM = "sum"
    COUNT = "cnt"
    CENTROIDS = "centroids"

class ImageSource(str, Enum):
    """Image source types"""
    UPLOAD = "upload"
    S3_PATH = "s3_path"
    URL = "url"

class ProcessingMode(str, Enum):
    """Processing mode options"""
    STANDARD = "standard"
    FAST = "fast"
    HIGH_QUALITY = "high_quality"

class BuildingExtractionRequest(BaseModel):
    """Building extraction request model"""
    image_source: ImageSource = Field(..., description="Source of the image")
    image_path: Optional[str] = Field(None, description="Path to image (S3 key or URL)")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for detection")
    apply_regularization: bool = Field(True, description="Apply geometric regularization")
    processing_mode: ProcessingMode = Field(ProcessingMode.STANDARD, description="Processing mode")
    tile_size: int = Field(512, ge=256, le=2048, description="Tile size for processing")
    overlap_ratio: float = Field(0.1, ge=0.0, le=0.5, description="Overlap ratio for tiling")
    
    @validator('image_path')
    def validate_image_path(cls, v, values):
        source = values.get('image_source')
        if source in [ImageSource.S3_PATH, ImageSource.URL] and not v:
            raise ValueError(f"image_path is required when image_source is {source}")
        return v

class StateProcessingRequest(BaseModel):
    """State processing request model"""
    state_name: str = Field(..., description="US state name (e.g., 'California')")
    layer_type: LayerType = Field(LayerType.AVG, description="Type of raster layer to process")
    tile_size: int = Field(512, ge=256, le=2048, description="Tile size for processing")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size for ML processing")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    apply_regularization: bool = Field(True, description="Apply geometric regularization")
    processing_mode: ProcessingMode = Field(ProcessingMode.STANDARD, description="Processing mode")
    
    @validator('state_name')
    def validate_state_name(cls, v):
        # List of valid US state names
        valid_states = {
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
            'Connecticut', 'Delaware', 'DistrictofColumbia', 'Florida', 'Georgia',
            'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
            'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
            'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'NewHampshire',
            'NewJersey', 'NewMexico', 'NewYork', 'NorthCarolina', 'NorthDakota',
            'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'RhodeIsland', 'SouthCarolina',
            'SouthDakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
            'Washington', 'WestVirginia', 'Wisconsin', 'Wyoming'
        }
        
        if v not in valid_states:
            raise ValueError(f"Invalid state name. Must be one of: {', '.join(sorted(valid_states))}")
        
        return v

class BuildingExtractionResponse(BaseModel):
    """Building extraction response model"""
    job_id: int
    task_id: str
    detected_buildings: int = 0
    processing_time_seconds: Optional[float] = None
    confidence_scores: List[float] = []
    building_polygons: List[List[Tuple[float, float]]] = []
    metadata: Dict[str, Any] = {}

class ProcessingProgress(BaseModel):
    """Processing progress model"""
    current_step: str
    total_steps: int
    completed_steps: int
    progress_percentage: float = Field(ge=0.0, le=100.0)
    estimated_time_remaining: Optional[int] = None  # seconds
    current_tile: Optional[int] = None
    total_tiles: Optional[int] = None

class ModelPrediction(BaseModel):
    """Individual model prediction"""
    confidence: float = Field(ge=0.0, le=1.0)
    polygon: List[Tuple[float, float]]
    area: float = Field(ge=0.0, description="Building area in square units")
    bbox: Tuple[float, float, float, float] = Field(description="Bounding box (x1, y1, x2, y2)")
    
class ProcessingResults(BaseModel):
    """Complete processing results"""
    job_id: int
    status: str
    total_buildings: int
    predictions: List[ModelPrediction]
    processing_metadata: Dict[str, Any] = {}
    quality_metrics: Dict[str, float] = {}
    
class ModelConfiguration(BaseModel):
    """ML model configuration"""
    model_name: str = Field("mask_rcnn", description="Model architecture name")
    model_version: str = Field("v1.0", description="Model version")
    input_size: Tuple[int, int] = Field((512, 512), description="Model input size (width, height)")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Non-maximum suppression threshold")
    max_detections: int = Field(100, ge=1, le=1000, description="Maximum detections per image")

class RegularizationSettings(BaseModel):
    """Geometric regularization settings"""
    enabled: bool = Field(True, description="Enable regularization")
    simplify_tolerance: float = Field(0.1, ge=0.0, le=10.0, description="Polygon simplification tolerance")
    min_area: float = Field(10.0, ge=0.0, description="Minimum building area")
    max_area: float = Field(100000.0, ge=0.0, description="Maximum building area")
    aspect_ratio_threshold: float = Field(10.0, ge=1.0, description="Maximum aspect ratio")

class ProcessingConfig(BaseModel):
    """Complete processing configuration"""
    model_config: ModelConfiguration = ModelConfiguration()
    regularization: RegularizationSettings = RegularizationSettings()
    tile_size: int = Field(512, ge=256, le=2048)
    overlap_ratio: float = Field(0.1, ge=0.0, le=0.5)
    batch_size: int = Field(32, ge=1, le=128)
    use_gpu: bool = Field(True, description="Use GPU acceleration if available")

class ValidationResults(BaseModel):
    """Model validation results"""
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    map_score: float = Field(ge=0.0, le=1.0, description="Mean Average Precision")
    validation_loss: float = Field(ge=0.0)
    
class ModelPerformance(BaseModel):
    """Model performance metrics"""
    inference_time_ms: float = Field(ge=0.0, description="Average inference time per image")
    throughput_fps: float = Field(ge=0.0, description="Frames per second")
    memory_usage_mb: float = Field(ge=0.0, description="Memory usage in MB")
    gpu_utilization: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU utilization percentage")