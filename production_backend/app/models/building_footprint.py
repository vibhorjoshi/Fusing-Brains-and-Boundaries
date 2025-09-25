"""
Building footprint data models
Core models for storing extracted building footprints and analysis results
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, Float, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import enum
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

from app.core.database import Base

class BuildingType(str, enum.Enum):
    """Building type classification"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    INSTITUTIONAL = "institutional"
    MIXED_USE = "mixed_use"
    UNKNOWN = "unknown"

class ExtractionMethod(str, enum.Enum):
    """Method used for building extraction"""
    MASK_RCNN = "mask_rcnn"
    MANUAL = "manual"
    IMPORTED = "imported"
    HYBRID = "hybrid"

class QualityScore(str, enum.Enum):
    """Quality assessment of extracted buildings"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 80-89%
    FAIR = "fair"           # 70-79%
    POOR = "poor"           # <70%

class BuildingFootprint(Base):
    """Main building footprint model"""
    __tablename__ = "building_footprints"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    
    # Geometric data
    geometry = Column(Text, nullable=False)  # GeoJSON or WKT polygon
    centroid_lat = Column(Float, nullable=False, index=True)
    centroid_lon = Column(Float, nullable=False, index=True)
    area_sqm = Column(Float, nullable=False, index=True)
    perimeter_m = Column(Float, nullable=False)
    
    # Geographic location
    country = Column(String(100), nullable=True, index=True)
    state = Column(String(100), nullable=True, index=True)
    city = Column(String(100), nullable=True, index=True)
    address = Column(Text, nullable=True)
    postal_code = Column(String(20), nullable=True)
    
    # Building characteristics
    building_type = Column(Enum(BuildingType), default=BuildingType.UNKNOWN, index=True)
    estimated_height_m = Column(Float, nullable=True)
    estimated_floors = Column(Integer, nullable=True)
    roof_type = Column(String(50), nullable=True)
    
    # Extraction metadata
    extraction_method = Column(Enum(ExtractionMethod), default=ExtractionMethod.MASK_RCNN, nullable=False)
    confidence_score = Column(Float, nullable=True)  # ML confidence 0-1
    quality_score = Column(Enum(QualityScore), nullable=True)
    
    # Processing metadata
    source_image_id = Column(String(255), nullable=True)  # Reference to source satellite image
    processing_job_id = Column(Integer, ForeignKey("processing_jobs.id"), nullable=True)
    regularized = Column(Boolean, default=False)  # Was geometric regularization applied
    
    # Validation and review
    reviewed_by_user = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    validation_status = Column(String(20), default="pending")  # pending, approved, rejected
    validation_notes = Column(Text, nullable=True)
    
    # Additional attributes (flexible storage)
    attributes = Column(JSON, nullable=True)  # Additional building attributes
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    processing_job = relationship("ProcessingJob", back_populates="building_footprints")
    reviewed_by = relationship("User")
    quality_metrics = relationship("BuildingQualityMetrics", back_populates="building_footprint")
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert building footprint to GeoJSON format"""
        return {
            "type": "Feature",
            "id": str(self.uuid),
            "geometry": eval(self.geometry) if isinstance(self.geometry, str) else self.geometry,
            "properties": {
                "id": self.id,
                "uuid": str(self.uuid),
                "area_sqm": self.area_sqm,
                "perimeter_m": self.perimeter_m,
                "building_type": self.building_type.value if self.building_type else None,
                "estimated_height_m": self.estimated_height_m,
                "estimated_floors": self.estimated_floors,
                "confidence_score": self.confidence_score,
                "quality_score": self.quality_score.value if self.quality_score else None,
                "extraction_method": self.extraction_method.value,
                "regularized": self.regularized,
                "country": self.country,
                "state": self.state,
                "city": self.city,
                "address": self.address,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "attributes": self.attributes
            }
        }

class BuildingQualityMetrics(Base):
    """Quality metrics for extracted building footprints"""
    __tablename__ = "building_quality_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    building_footprint_id = Column(Integer, ForeignKey("building_footprints.id"), nullable=False)
    
    # Geometric quality metrics
    iou_score = Column(Float, nullable=True)  # IoU with ground truth (if available)
    hausdorff_distance = Column(Float, nullable=True)
    geometric_accuracy = Column(Float, nullable=True)
    
    # Shape analysis
    rectangularity_score = Column(Float, nullable=True)  # How rectangular the building is
    corner_count = Column(Integer, nullable=True)
    right_angle_count = Column(Integer, nullable=True)
    shape_complexity = Column(Float, nullable=True)
    
    # Regularization metrics (if applied)
    pre_regularization_area = Column(Float, nullable=True)
    post_regularization_area = Column(Float, nullable=True)
    area_change_percent = Column(Float, nullable=True)
    shape_preservation_score = Column(Float, nullable=True)
    
    # Detection confidence metrics
    mask_confidence = Column(Float, nullable=True)
    bbox_confidence = Column(Float, nullable=True)
    segmentation_quality = Column(Float, nullable=True)
    
    # Comparison with ground truth (if available)
    ground_truth_available = Column(Boolean, default=False)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Additional metrics
    processing_time_ms = Column(Integer, nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now())
    
    # Relationship
    building_footprint = relationship("BuildingFootprint", back_populates="quality_metrics")

class BuildingCluster(Base):
    """Groups of related buildings (e.g., housing developments, commercial complexes)"""
    __tablename__ = "building_clusters"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    
    # Cluster information
    name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    cluster_type = Column(String(50), nullable=True)  # development, complex, district
    
    # Geographic bounds
    bbox_north = Column(Float, nullable=False)
    bbox_south = Column(Float, nullable=False)
    bbox_east = Column(Float, nullable=False)
    bbox_west = Column(Float, nullable=False)
    
    # Cluster statistics
    building_count = Column(Integer, nullable=False, default=0)
    total_area_sqm = Column(Float, nullable=False, default=0)
    avg_building_area = Column(Float, nullable=True)
    density_per_sqkm = Column(Float, nullable=True)
    
    # Processing metadata
    processing_job_id = Column(Integer, ForeignKey("processing_jobs.id"), nullable=True)
    auto_generated = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class BuildingHistory(Base):
    """Track changes to building footprints over time"""
    __tablename__ = "building_history"
    
    id = Column(Integer, primary_key=True, index=True)
    building_footprint_id = Column(Integer, ForeignKey("building_footprints.id"), nullable=False)
    
    # Change information
    change_type = Column(String(20), nullable=False)  # created, updated, deleted, merged, split
    changed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    change_reason = Column(String(200), nullable=True)
    
    # Previous values (JSON storage)
    previous_geometry = Column(Text, nullable=True)
    previous_attributes = Column(JSON, nullable=True)
    
    # New values
    new_geometry = Column(Text, nullable=True)
    new_attributes = Column(JSON, nullable=True)
    
    # Metadata
    change_notes = Column(Text, nullable=True)
    automated_change = Column(Boolean, default=False)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    building_footprint = relationship("BuildingFootprint")
    user = relationship("User")