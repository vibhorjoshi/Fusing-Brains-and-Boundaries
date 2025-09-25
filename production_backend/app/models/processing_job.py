"""
Processing job models
Track ML inference jobs, batch processing, and task management
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, Float, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import uuid

from app.core.database import Base

class JobType(str, enum.Enum):
    """Type of processing job"""
    SINGLE_IMAGE = "single_image"
    BATCH_PROCESSING = "batch_processing"
    STATE_PROCESSING = "state_processing"
    EVALUATION = "evaluation"
    TRAINING = "training"
    INFERENCE = "inference"

class JobStatus(str, enum.Enum):
    """Processing job status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class JobPriority(str, enum.Enum):
    """Job priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class ProcessingJob(Base):
    """Main processing job model for ML operations"""
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Job identification
    job_name = Column(String(200), nullable=True)
    job_type = Column(Enum(JobType), nullable=False, index=True)
    job_description = Column(Text, nullable=True)
    
    # Job ownership and permissions
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    organization = Column(String(200), nullable=True)
    
    # Job status and lifecycle
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    priority = Column(Enum(JobPriority), default=JobPriority.NORMAL, nullable=False)
    
    # Timing information
    created_at = Column(DateTime, default=func.now(), index=True)
    queued_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    estimated_duration_minutes = Column(Integer, nullable=True)
    
    # Progress tracking
    progress_percent = Column(Float, default=0.0, nullable=False)
    current_step = Column(String(200), nullable=True)
    total_steps = Column(Integer, nullable=True)
    
    # Input parameters
    input_parameters = Column(JSON, nullable=True)  # ML model parameters, thresholds, etc.
    input_files = Column(JSON, nullable=True)  # List of input file IDs/paths
    
    # Processing configuration
    ml_model_version = Column(String(50), nullable=True)
    confidence_threshold = Column(Float, default=0.5)
    apply_regularization = Column(Boolean, default=True)
    batch_size = Column(Integer, default=4)
    
    # Geographic scope
    geographic_bounds = Column(JSON, nullable=True)  # Bounding box or region
    target_states = Column(JSON, nullable=True)  # List of US states to process
    coordinate_system = Column(String(50), default="EPSG:4326")
    
    # Results and output
    output_files = Column(JSON, nullable=True)  # Generated output files
    results_summary = Column(JSON, nullable=True)  # Summary statistics
    buildings_extracted = Column(Integer, default=0)
    total_area_processed_sqkm = Column(Float, nullable=True)
    
    # Performance metrics
    processing_time_seconds = Column(Float, nullable=True)
    memory_used_gb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    gpu_usage_percent = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # AWS integration
    aws_job_id = Column(String(200), nullable=True)  # AWS Batch job ID or SQS message ID
    s3_input_path = Column(String(500), nullable=True)
    s3_output_path = Column(String(500), nullable=True)
    
    # Quality metrics
    average_confidence = Column(Float, nullable=True)
    average_iou = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="processing_jobs")
    building_footprints = relationship("BuildingFootprint", back_populates="processing_job")
    job_logs = relationship("JobLog", back_populates="job")
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate job duration in minutes"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() / 60
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if job is currently active"""
        return self.status in [JobStatus.QUEUED, JobStatus.RUNNING]
    
    @property
    def is_finished(self) -> bool:
        """Check if job has finished (success or failure)"""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    def update_progress(self, percent: float, step: str = None):
        """Update job progress"""
        self.progress_percent = min(100.0, max(0.0, percent))
        if step:
            self.current_step = step
    
    def mark_started(self):
        """Mark job as started"""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.queued_at = self.queued_at or datetime.utcnow()
    
    def mark_completed(self, results: Dict[str, Any] = None):
        """Mark job as completed with optional results"""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress_percent = 100.0
        if results:
            self.results_summary = results
    
    def mark_failed(self, error_message: str, error_details: Dict[str, Any] = None):
        """Mark job as failed with error information"""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if error_details:
            self.error_details = error_details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API responses"""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "job_name": self.job_name,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "buildings_extracted": self.buildings_extracted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_minutes": self.duration_minutes,
            "error_message": self.error_message,
            "results_summary": self.results_summary
        }

class JobLog(Base):
    """Detailed logging for processing jobs"""
    __tablename__ = "job_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("processing_jobs.id"), nullable=False, index=True)
    
    # Log entry details
    log_level = Column(String(10), nullable=False)  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    component = Column(String(50), nullable=True)  # Which component logged this
    
    # Additional context
    step_number = Column(Integer, nullable=True)
    progress_percent = Column(Float, nullable=True)
    
    # Performance data
    memory_usage_mb = Column(Float, nullable=True)
    cpu_percent = Column(Float, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationship
    job = relationship("ProcessingJob", back_populates="job_logs")

class BatchOperation(Base):
    """Track batch operations across multiple jobs"""
    __tablename__ = "batch_operations"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Batch information
    batch_name = Column(String(200), nullable=False)
    batch_description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Batch configuration
    job_type = Column(Enum(JobType), nullable=False)
    total_jobs = Column(Integer, nullable=False, default=0)
    completed_jobs = Column(Integer, nullable=False, default=0)
    failed_jobs = Column(Integer, nullable=False, default=0)
    
    # Status
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    
    # Timing
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Results
    total_buildings_extracted = Column(Integer, default=0)
    total_area_processed_sqkm = Column(Float, default=0.0)
    
    # AWS integration
    aws_batch_job_queue = Column(String(200), nullable=True)
    
    @property
    def progress_percent(self) -> float:
        """Calculate batch progress percentage"""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs / self.total_jobs) * 100.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of completed jobs"""
        completed = self.completed_jobs + self.failed_jobs
        if completed == 0:
            return 0.0
        return (self.completed_jobs / completed) * 100.0

class JobQueue(Base):
    """Job queue management for task scheduling"""
    __tablename__ = "job_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("processing_jobs.id"), nullable=False, unique=True)
    
    # Queue position and priority
    queue_position = Column(Integer, nullable=False, index=True)
    estimated_wait_minutes = Column(Integer, nullable=True)
    
    # Resource requirements
    required_memory_gb = Column(Float, default=8.0)
    required_cpu_cores = Column(Integer, default=2)
    requires_gpu = Column(Boolean, default=True)
    
    # Scheduling
    scheduled_at = Column(DateTime, nullable=True)
    worker_node = Column(String(100), nullable=True)
    
    # Timestamps
    queued_at = Column(DateTime, default=func.now())
    dequeued_at = Column(DateTime, nullable=True)
    
    # Relationship
    job = relationship("ProcessingJob")