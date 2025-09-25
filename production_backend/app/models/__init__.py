"""
Model registry - Import all models for SQLAlchemy
"""

from .user import User, APIUsage, UserSession
from .building_footprint import (
    BuildingFootprint, 
    BuildingQualityMetrics, 
    BuildingCluster, 
    BuildingHistory
)
from .processing_job import (
    ProcessingJob, 
    JobLog, 
    BatchOperation, 
    JobQueue
)
from .file_storage import (
    FileStorage, 
    FileDownloadLog, 
    FileProcessingQueue, 
    S3Configuration
)

__all__ = [
    # User models
    "User",
    "APIUsage", 
    "UserSession",
    
    # Building footprint models
    "BuildingFootprint",
    "BuildingQualityMetrics",
    "BuildingCluster", 
    "BuildingHistory",
    
    # Processing job models
    "ProcessingJob",
    "JobLog",
    "BatchOperation",
    "JobQueue",
    
    # File storage models
    "FileStorage",
    "FileDownloadLog", 
    "FileProcessingQueue",
    "S3Configuration"
]