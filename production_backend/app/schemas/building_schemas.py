"""
Building footprint schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

class GeometryType(str, Enum):
    """Supported geometry types"""
    POLYGON = "Polygon"
    MULTIPOLYGON = "MultiPolygon"

class ExtractionMethod(str, Enum):
    """Building extraction methods"""
    MASK_RCNN = "mask_rcnn"
    MASK_RCNN_STATE = "mask_rcnn_state"
    MANUAL = "manual"
    IMPORTED = "imported"

class BuildingFootprintBase(BaseModel):
    """Base building footprint model"""
    geometry_type: GeometryType = Field(GeometryType.POLYGON, description="Geometry type")
    coordinates: List[List[Tuple[float, float]]] = Field(..., description="Polygon coordinates")
    area: Optional[float] = Field(None, ge=0.0, description="Building area in square units")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="ML confidence score")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality assessment score")

class BuildingFootprintCreate(BuildingFootprintBase):
    """Building footprint creation model"""
    state_name: Optional[str] = Field(None, max_length=50, description="US state name")
    county_name: Optional[str] = Field(None, max_length=100, description="County name")
    city_name: Optional[str] = Field(None, max_length=100, description="City name")
    extraction_method: ExtractionMethod = Field(ExtractionMethod.MASK_RCNN, description="Extraction method")
    regularized: bool = Field(False, description="Whether geometry was regularized")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Coordinates cannot be empty")
        
        for ring in v:
            if len(ring) < 3:
                raise ValueError("Polygon rings must have at least 3 points")
            
            # Check if polygon is closed (first and last points should be the same)
            if ring[0] != ring[-1]:
                ring.append(ring[0])  # Auto-close polygon
        
        return v

class BuildingFootprintUpdate(BaseModel):
    """Building footprint update model"""
    coordinates: Optional[List[List[Tuple[float, float]]]] = None
    area: Optional[float] = Field(None, ge=0.0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    state_name: Optional[str] = Field(None, max_length=50)
    county_name: Optional[str] = Field(None, max_length=100)
    city_name: Optional[str] = Field(None, max_length=100)
    regularized: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

class BuildingFootprintResponse(BuildingFootprintBase):
    """Building footprint response model"""
    id: int
    uuid: str
    job_id: Optional[int] = None
    user_id: int
    state_name: Optional[str] = None
    county_name: Optional[str] = None
    city_name: Optional[str] = None
    extraction_method: Optional[str] = None
    regularized: bool = False
    layer_type: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

class BuildingFootprintFilter(BaseModel):
    """Building footprint filtering model"""
    state_name: Optional[str] = None
    county_name: Optional[str] = None
    city_name: Optional[str] = None
    extraction_method: Optional[ExtractionMethod] = None
    confidence_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_max: Optional[float] = Field(None, ge=0.0, le=1.0)
    area_min: Optional[float] = Field(None, ge=0.0)
    area_max: Optional[float] = Field(None, ge=0.0)
    quality_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    regularized: Optional[bool] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None

class BuildingStatistics(BaseModel):
    """Building statistics model"""
    total_buildings: int = 0
    average_area: float = 0.0
    average_confidence: float = 0.0
    average_quality: float = 0.0
    state_distribution: Dict[str, int] = {}
    extraction_methods: Dict[str, int] = {}
    quality_metrics: Dict[str, Any] = {}

class BuildingSearch(BaseModel):
    """Building search model"""
    query: Optional[str] = Field(None, description="Search query")
    filters: Optional[BuildingFootprintFilter] = None
    bbox: Optional[Tuple[float, float, float, float]] = Field(None, description="Bounding box (min_x, min_y, max_x, max_y)")
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=1000, description="Page size")
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", regex="^(asc|desc)$", description="Sort order")

class BuildingExport(BaseModel):
    """Building export model"""
    format: str = Field("geojson", regex="^(geojson|shapefile|csv|kml)$")
    filters: Optional[BuildingFootprintFilter] = None
    include_metadata: bool = Field(True, description="Include metadata in export")
    coordinate_system: str = Field("EPSG:4326", description="Output coordinate system")

class BuildingValidation(BaseModel):
    """Building validation results"""
    is_valid: bool = True
    errors: List[str] = []
    warnings: List[str] = []
    area_calculated: Optional[float] = None
    perimeter: Optional[float] = None
    shape_metrics: Optional[Dict[str, float]] = None

class BuildingGeometry(BaseModel):
    """Simplified building geometry model"""
    type: GeometryType
    coordinates: List[List[Tuple[float, float]]]
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON format"""
        return {
            "type": self.type.value,
            "coordinates": self.coordinates
        }

class BuildingCollection(BaseModel):
    """Collection of buildings with metadata"""
    buildings: List[BuildingFootprintResponse]
    total_count: int
    page: int = 1
    page_size: int = 20
    has_next: bool = False
    has_previous: bool = False
    
class BuildingBatch(BaseModel):
    """Batch building operations model"""
    buildings: List[BuildingFootprintCreate] = Field(..., max_items=1000)
    operation: str = Field("create", regex="^(create|update|delete)$")
    validation_mode: str = Field("strict", regex="^(strict|lenient)$")
    
    @validator('buildings')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch must contain at least one building")
        if len(v) > 1000:
            raise ValueError("Batch cannot contain more than 1000 buildings")
        return v