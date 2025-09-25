"""
ML Processing API endpoints
Production endpoints for building extraction and ML operations
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging

from app.core.database import get_db
from app.models.processing_job import ProcessingJob, JobType, JobStatus
from app.models.user import User
from app.services.s3_service import s3_service
from app.services.sqs_service import sqs_service
from app.tasks.ml_processing import extract_buildings_from_image, process_state_data
from app.api.deps import get_current_user
from app.schemas.job_schemas import ProcessingJobCreate, ProcessingJobResponse
from app.schemas.ml_schemas import BuildingExtractionRequest, StateProcessingRequest

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/extract-buildings", response_model=ProcessingJobResponse)
async def extract_buildings_endpoint(
    request: BuildingExtractionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Extract building footprints from satellite imagery
    
    This endpoint creates a processing job for building extraction using Mask R-CNN
    and queues it for background processing.
    """
    try:
        # Create processing job
        job = ProcessingJob(
            job_name=f"Building extraction - {current_user.username}",
            job_type=JobType.SINGLE_IMAGE,
            job_description="Extract building footprints using Mask R-CNN",
            user_id=current_user.id,
            input_parameters={
                "confidence_threshold": request.confidence_threshold,
                "apply_regularization": request.apply_regularization,
                "image_source": request.image_source
            },
            confidence_threshold=request.confidence_threshold,
            apply_regularization=request.apply_regularization
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Queue job for processing
        task_result = extract_buildings_from_image.delay(
            job_id=job.id,
            confidence_threshold=request.confidence_threshold,
            apply_regularization=request.apply_regularization
        )
        
        # Update job with task ID
        job.aws_job_id = task_result.id
        job.status = JobStatus.QUEUED
        db.commit()
        
        logger.info(f"✅ Building extraction job created: {job.uuid}")
        
        return ProcessingJobResponse(
            id=job.id,
            uuid=job.uuid,
            job_type=job.job_type,
            status=job.status,
            created_at=job.created_at,
            task_id=task_result.id
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to create building extraction job: {e}")
        raise HTTPException(status_code=500, detail="Failed to create processing job")

@router.post("/process-state", response_model=ProcessingJobResponse)
async def process_state_endpoint(
    request: StateProcessingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process building footprint data for a specific US state
    
    This endpoint processes satellite imagery data for an entire US state,
    extracting building footprints at scale.
    """
    try:
        # Create processing job
        job = ProcessingJob(
            job_name=f"State processing: {request.state_name}",
            job_type=JobType.STATE_PROCESSING,
            job_description=f"Process building footprints for {request.state_name}",
            user_id=current_user.id,
            input_parameters={
                "state_name": request.state_name,
                "layer_type": request.layer_type,
                "tile_size": request.tile_size,
                "batch_size": request.batch_size
            },
            target_states=[request.state_name],
            batch_size=request.batch_size
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Queue job for processing
        task_result = process_state_data.delay(
            job_id=job.id,
            state_name=request.state_name,
            layer_type=request.layer_type
        )
        
        # Update job with task ID
        job.aws_job_id = task_result.id
        job.status = JobStatus.QUEUED
        db.commit()
        
        logger.info(f"✅ State processing job created: {job.uuid} for {request.state_name}")
        
        return ProcessingJobResponse(
            id=job.id,
            uuid=job.uuid,
            job_type=job.job_type,
            status=job.status,
            created_at=job.created_at,
            task_id=task_result.id
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to create state processing job: {e}")
        raise HTTPException(status_code=500, detail="Failed to create processing job")

@router.post("/batch-process")
async def batch_process_states(
    state_list: List[str],
    layer_type: str = "avg",
    max_states: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Batch process multiple states for building footprint extraction
    
    Creates multiple processing jobs for efficient large-scale processing.
    """
    try:
        if len(state_list) > max_states:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many states requested. Maximum allowed: {max_states}"
            )
        
        created_jobs = []
        
        for state_name in state_list:
            # Create individual processing job for each state
            job = ProcessingJob(
                job_name=f"Batch processing: {state_name}",
                job_type=JobType.BATCH_PROCESSING,
                job_description=f"Batch processing for {state_name}",
                user_id=current_user.id,
                input_parameters={
                    "state_name": state_name,
                    "layer_type": layer_type,
                    "batch_processing": True
                },
                target_states=[state_name]
            )
            
            db.add(job)
            db.commit()
            db.refresh(job)
            
            # Queue job
            task_result = process_state_data.delay(
                job_id=job.id,
                state_name=state_name,
                layer_type=layer_type
            )
            
            job.aws_job_id = task_result.id
            job.status = JobStatus.QUEUED
            
            created_jobs.append({
                "state": state_name,
                "job_id": job.id,
                "job_uuid": job.uuid,
                "task_id": task_result.id
            })
        
        db.commit()
        
        logger.info(f"✅ Batch processing created {len(created_jobs)} jobs")
        
        return {
            "status": "success",
            "message": f"Created {len(created_jobs)} processing jobs",
            "jobs": created_jobs
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create batch processing jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to create batch processing jobs")

@router.get("/task-status/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of a background processing task
    
    Returns real-time status and progress information for a Celery task.
    """
    try:
        from app.core.celery_app import get_task_info
        
        task_info = get_task_info(task_id)
        
        return {
            "task_id": task_id,
            "status": task_info.get("status"),
            "result": task_info.get("result"),
            "ready": task_info.get("ready"),
            "successful": task_info.get("successful"),
            "failed": task_info.get("failed")
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")

@router.post("/upload-image")
async def upload_image_for_processing(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    apply_regularization: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload satellite image and queue for building extraction
    
    This endpoint handles file upload to S3 and creates a processing job.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        
        # Upload to S3
        from app.models.file_storage import FileType
        
        file_record = await s3_service.upload_file(
            file_data=file.file,
            filename=file.filename,
            file_type=FileType.SATELLITE_IMAGE,
            user_id=current_user.id,
            metadata={
                "uploaded_for_processing": True,
                "confidence_threshold": confidence_threshold,
                "apply_regularization": apply_regularization
            }
        )
        
        db.add(file_record)
        db.commit()
        db.refresh(file_record)
        
        # Create processing job
        job = ProcessingJob(
            job_name=f"Process uploaded image: {file.filename}",
            job_type=JobType.SINGLE_IMAGE,
            job_description=f"Process uploaded satellite image: {file.filename}",
            user_id=current_user.id,
            input_parameters={
                "file_id": file_record.id,
                "confidence_threshold": confidence_threshold,
                "apply_regularization": apply_regularization
            },
            input_files=[file_record.id],
            s3_input_path=file_record.storage_path
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Queue for processing
        task_result = extract_buildings_from_image.delay(
            job_id=job.id,
            confidence_threshold=confidence_threshold,
            apply_regularization=apply_regularization
        )
        
        job.aws_job_id = task_result.id
        job.status = JobStatus.QUEUED
        db.commit()
        
        logger.info(f"✅ Image uploaded and queued for processing: {file.filename}")
        
        return {
            "status": "success",
            "file_id": file_record.id,
            "job_id": job.id,
            "job_uuid": job.uuid,
            "task_id": task_result.id,
            "message": "Image uploaded and queued for processing"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to upload and process image: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload and process image")

@router.get("/processing-stats")
async def get_processing_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get processing statistics and system status
    
    Returns comprehensive statistics about ML processing jobs and system health.
    """
    try:
        from sqlalchemy import func
        from datetime import datetime, timedelta
        
        # Get job statistics
        total_jobs = db.query(ProcessingJob).count()
        
        completed_jobs = db.query(ProcessingJob).filter(
            ProcessingJob.status == JobStatus.COMPLETED
        ).count()
        
        failed_jobs = db.query(ProcessingJob).filter(
            ProcessingJob.status == JobStatus.FAILED
        ).count()
        
        running_jobs = db.query(ProcessingJob).filter(
            ProcessingJob.status == JobStatus.RUNNING
        ).count()
        
        queued_jobs = db.query(ProcessingJob).filter(
            ProcessingJob.status == JobStatus.QUEUED
        ).count()
        
        # Get recent activity (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_jobs = db.query(ProcessingJob).filter(
            ProcessingJob.created_at >= yesterday
        ).count()
        
        # Get building statistics
        from app.models.building_footprint import BuildingFootprint
        
        total_buildings = db.query(BuildingFootprint).count()
        
        recent_buildings = db.query(BuildingFootprint).filter(
            BuildingFootprint.created_at >= yesterday
        ).count()
        
        # Get Celery worker stats
        from app.core.celery_app import get_worker_stats
        worker_stats = get_worker_stats()
        
        return {
            "processing_statistics": {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "running_jobs": running_jobs,
                "queued_jobs": queued_jobs,
                "success_rate": (completed_jobs / max(total_jobs, 1)) * 100,
                "recent_jobs_24h": recent_jobs
            },
            "building_statistics": {
                "total_buildings_extracted": total_buildings,
                "recent_buildings_24h": recent_buildings
            },
            "system_status": {
                "celery_workers": worker_stats,
                "queue_health": "operational" if running_jobs + queued_jobs < 100 else "busy"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get processing stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve processing statistics")