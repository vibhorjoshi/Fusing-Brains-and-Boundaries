"""
File storage models
AWS S3 integration and file management for satellite imagery and results
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, Float, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from typing import Optional, Dict, Any
import hashlib
from datetime import datetime
import mimetypes

from app.core.database import Base

class FileType(str, enum.Enum):
    """Types of files stored in the system"""
    SATELLITE_IMAGE = "satellite_image"
    AERIAL_IMAGE = "aerial_image"
    GEOTIFF = "geotiff"
    SHAPEFILE = "shapefile"
    GEOJSON = "geojson"
    CSV = "csv"
    ZIP_ARCHIVE = "zip_archive"
    ML_MODEL = "ml_model"
    RESULTS = "results"
    LOGS = "logs"
    REPORT = "report"

class FileStatus(str, enum.Enum):
    """File processing and storage status"""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"

class StorageProvider(str, enum.Enum):
    """Storage provider types"""
    AWS_S3 = "aws_s3"
    LOCAL = "local"
    AZURE_BLOB = "azure_blob"
    GCP_STORAGE = "gcp_storage"

class FileStorage(Base):
    """File storage and metadata management"""
    __tablename__ = "file_storage"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # File identification
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_extension = Column(String(10), nullable=True)
    mime_type = Column(String(100), nullable=True)
    
    # File properties
    file_size_bytes = Column(Integer, nullable=False)
    file_hash_md5 = Column(String(32), nullable=True, index=True)
    file_hash_sha256 = Column(String(64), nullable=True, index=True)
    
    # File classification
    file_type = Column(Enum(FileType), nullable=False, index=True)
    status = Column(Enum(FileStatus), default=FileStatus.UPLOADING, nullable=False)
    
    # Storage information
    storage_provider = Column(Enum(StorageProvider), default=StorageProvider.AWS_S3, nullable=False)
    storage_path = Column(String(500), nullable=False)  # S3 key or file path
    storage_bucket = Column(String(100), nullable=True)
    storage_region = Column(String(50), nullable=True)
    
    # Access control
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    is_public = Column(Boolean, default=False)
    access_permissions = Column(JSON, nullable=True)  # Fine-grained permissions
    
    # Geographic metadata (for geographic files)
    geographic_bounds = Column(JSON, nullable=True)  # Bounding box
    coordinate_system = Column(String(50), nullable=True)  # CRS/EPSG code
    resolution_meters = Column(Float, nullable=True)  # Spatial resolution
    
    # Image-specific metadata
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    image_bands = Column(Integer, nullable=True)  # Number of spectral bands
    image_format = Column(String(20), nullable=True)  # TIFF, PNG, JPEG, etc.
    
    # Processing metadata
    processing_job_id = Column(Integer, ForeignKey("processing_jobs.id"), nullable=True)
    processed_by_user = Column(Integer, ForeignKey("users.id"), nullable=True)
    processing_parameters = Column(JSON, nullable=True)
    
    # File relationships
    parent_file_id = Column(Integer, ForeignKey("file_storage.id"), nullable=True)  # Derived files
    is_derived = Column(Boolean, default=False)
    derivation_method = Column(String(100), nullable=True)  # How was this file created
    
    # Metadata and tags
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # List of tags for organization
    metadata = Column(JSON, nullable=True)  # Additional file metadata
    
    # Lifecycle management
    expires_at = Column(DateTime, nullable=True)  # Automatic deletion date
    archived_at = Column(DateTime, nullable=True)
    backup_location = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_accessed = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="file_uploads", foreign_keys=[user_id])
    processed_by = relationship("User", foreign_keys=[processed_by_user])
    processing_job = relationship("ProcessingJob")
    parent_file = relationship("FileStorage", remote_side=[id])
    derived_files = relationship("FileStorage", remote_side=[parent_file_id])
    download_logs = relationship("FileDownloadLog", back_populates="file")
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in MB"""
        return self.file_size_bytes / (1024 * 1024) if self.file_size_bytes else 0
    
    @property
    def s3_url(self) -> Optional[str]:
        """Get S3 URL for the file"""
        if self.storage_provider == StorageProvider.AWS_S3 and self.storage_bucket:
            region = self.storage_region or "us-east-1"
            return f"https://{self.storage_bucket}.s3.{region}.amazonaws.com/{self.storage_path}"
        return None
    
    def generate_hash(self, content: bytes) -> tuple:
        """Generate MD5 and SHA256 hashes for file content"""
        md5_hash = hashlib.md5(content).hexdigest()
        sha256_hash = hashlib.sha256(content).hexdigest()
        return md5_hash, sha256_hash
    
    def update_metadata_from_filename(self):
        """Extract metadata from filename"""
        self.file_extension = self.original_filename.split('.')[-1].lower() if '.' in self.original_filename else None
        self.mime_type, _ = mimetypes.guess_type(self.original_filename)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file to dictionary for API responses"""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_type": self.file_type.value,
            "status": self.status.value,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": self.file_size_mb,
            "mime_type": self.mime_type,
            "storage_provider": self.storage_provider.value,
            "is_public": self.is_public,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "geographic_bounds": self.geographic_bounds,
            "coordinate_system": self.coordinate_system,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "description": self.description
        }

class FileDownloadLog(Base):
    """Track file downloads for analytics and access control"""
    __tablename__ = "file_download_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("file_storage.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Download details
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    download_method = Column(String(20), default="api")  # api, direct, presigned
    
    # Download success/failure
    success = Column(Boolean, nullable=False)
    bytes_downloaded = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Performance metrics
    download_duration_seconds = Column(Float, nullable=True)
    
    # Timestamp
    downloaded_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    file = relationship("FileStorage", back_populates="download_logs")
    user = relationship("User")

class FileProcessingQueue(Base):
    """Queue for file processing operations"""
    __tablename__ = "file_processing_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("file_storage.id"), nullable=False, unique=True)
    
    # Processing configuration
    operation = Column(String(50), nullable=False)  # thumbnail, validation, conversion, etc.
    parameters = Column(JSON, nullable=True)
    priority = Column(Integer, default=5)  # 1=highest, 10=lowest
    
    # Status tracking
    status = Column(String(20), default="queued")  # queued, processing, completed, failed
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    last_attempt_at = Column(DateTime, nullable=True)
    
    # Scheduling
    scheduled_at = Column(DateTime, nullable=True)
    worker_id = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    
    # Relationship
    file = relationship("FileStorage")

class S3Configuration(Base):
    """S3 bucket configurations and settings"""
    __tablename__ = "s3_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Bucket information
    bucket_name = Column(String(100), nullable=False, unique=True)
    region = Column(String(50), nullable=False)
    
    # Access configuration
    access_key_id = Column(String(100), nullable=True)  # If not using IAM roles
    encryption_enabled = Column(Boolean, default=True)
    versioning_enabled = Column(Boolean, default=True)
    
    # Lifecycle policies
    default_storage_class = Column(String(50), default="STANDARD")
    transition_to_ia_days = Column(Integer, default=30)  # Transition to Infrequent Access
    transition_to_glacier_days = Column(Integer, default=90)
    expiration_days = Column(Integer, nullable=True)
    
    # Usage and limits
    max_file_size_gb = Column(Float, default=10.0)
    monthly_upload_limit_gb = Column(Float, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())