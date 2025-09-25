"""
Processing job schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from app.models.processing_job import JobType, JobStatus

class ProcessingJobBase(BaseModel):
    """Base processing job model"""
    job_name: str = Field(..., min_length=1, max_length=200, description="Job name")
    job_description: Optional[str] = Field(None, max_length=500, description="Job description")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold (0.0-1.0)")
    apply_regularization: Optional[bool] = Field(True, description="Apply regularization to results")

class ProcessingJobCreate(ProcessingJobBase):
    """Processing job creation model"""
    job_type: JobType = Field(..., description="Type of processing job")
    input_parameters: Optional[Dict[str, Any]] = Field(None, description="Job input parameters")
    target_states: Optional[List[str]] = Field(None, description="Target states for processing")
    batch_size: Optional[int] = Field(32, ge=1, le=128, description="Batch size for processing")

class ProcessingJobUpdate(BaseModel):
    """Processing job update model"""
    job_name: Optional[str] = Field(None, min_length=1, max_length=200)
    job_description: Optional[str] = Field(None, max_length=500)
    status: Optional[JobStatus] = Field(None, description="Job status")

class ProcessingJobResponse(BaseModel):
    """Processing job response model"""
    id: int
    uuid: str
    job_name: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: Optional[float] = None
    task_id: Optional[str] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True

class ProcessingJobDetail(ProcessingJobResponse):
    """Detailed processing job model"""
    job_description: Optional[str] = None
    input_parameters: Optional[Dict[str, Any]] = None
    output_results: Optional[Dict[str, Any]] = None
    target_states: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None
    apply_regularization: Optional[bool] = None
    batch_size: Optional[int] = None
    processing_time_minutes: Optional[float] = None
    input_files: Optional[List[int]] = None
    output_files: Optional[List[int]] = None
    s3_input_path: Optional[str] = None
    s3_output_path: Optional[str] = None
    aws_job_id: Optional[str] = None
    
class JobStatistics(BaseModel):
    """Job statistics model"""
    total_jobs: int = 0
    queued_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    success_rate: float = 0.0
    average_processing_time: Optional[float] = None
    
class JobFilter(BaseModel):
    """Job filtering model"""
    job_type: Optional[JobType] = None
    status: Optional[JobStatus] = None
    user_id: Optional[int] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    target_states: Optional[List[str]] = None
    
class JobSearch(BaseModel):
    """Job search model"""
    query: Optional[str] = Field(None, description="Search query")
    filters: Optional[JobFilter] = None
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Page size")
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", regex="^(asc|desc)$", description="Sort order")

class JobBatch(BaseModel):
    """Batch job creation model"""
    job_name_prefix: str = Field(..., description="Prefix for batch job names")
    job_description: Optional[str] = None
    jobs: List[Dict[str, Any]] = Field(..., description="List of job configurations")
    
    @validator('jobs')
    def validate_jobs(cls, v):
        if len(v) == 0:
            raise ValueError("Batch must contain at least one job")
        if len(v) > 50:
            raise ValueError("Batch cannot contain more than 50 jobs")
        return v