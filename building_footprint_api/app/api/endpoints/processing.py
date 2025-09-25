"""
Processing API endpoints
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from app.models.schemas import ProcessingRequest, ProcessingResult, JobProgress
from app.services.pipeline import PipelineService
from app.services.workflow import WorkflowManager
from app.utils.websocket_manager import ConnectionManager
from app.api.dependencies import get_pipeline_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Dependency injection
workflow_manager = WorkflowManager()
connection_manager = ConnectionManager()

@router.post("/process", response_model=Dict[str, str])
async def process_building_footprints(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    pipeline_service: PipelineService = Depends(get_pipeline_service)
):
    """
    Start building footprint processing job
    """
    try:
        # Validate request
        if not pipeline_service.mask_rcnn:
            await pipeline_service.initialize()
        
        # Start background processing
        background_tasks.add_task(
            process_in_background,
            request,
            connection_manager,
            pipeline_service
        )
        
        return {
            "job_id": request.job_id,
            "status": "processing",
            "message": "Job started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start processing job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get job processing status
    """
    try:
        status = await workflow_manager.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """
    Get job processing results
    """
    try:
        results = await workflow_manager.get_job_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="Results not found")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a processing job
    """
    try:
        success = await workflow_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {"message": "Job cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_in_background(
    request: ProcessingRequest, 
    connection_manager: ConnectionManager, 
    pipeline_service: PipelineService
):
    """Background task for processing"""
    
    async def progress_callback(job_id: str, stage: str, progress: float, message: str):
        """Send progress updates via WebSocket"""
        progress_update = {
            "job_id": job_id,
            "stage": stage,
            "progress": progress,
            "message": message
        }
        await connection_manager.broadcast(progress_update)
    
    try:
        result = await pipeline_service.process_request(request, progress_callback)
        await workflow_manager.save_job_results(request.job_id, result)
        
        # Send completion notification
        await connection_manager.broadcast({
            "job_id": request.job_id,
            "status": "completed",
            "message": "Processing completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Background processing failed: {str(e)}")
        await connection_manager.broadcast({
            "job_id": request.job_id,
            "status": "failed",
            "message": str(e)
        })