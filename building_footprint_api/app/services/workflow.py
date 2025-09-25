"""
Workflow Manager Service
"""

import asyncio
import uuid
import logging
import json
from typing import Dict, Any, Optional, Callable
import time
import redis

from app.models.schemas import ProcessingRequest, ProcessingResponse, JobStatus, ProcessingStage
from app.services.pipeline import PipelineService
from app.core.config import get_settings
from app.core.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)
settings = get_settings()

class WorkflowManager:
    """
    Workflow Manager for building extraction jobs
    
    Handles:
    - Job creation and management
    - Status tracking
    - Callback notifications
    - Results storage
    """
    
    def __init__(self, pipeline_service: Optional[PipelineService] = None, websocket_manager: Optional[WebSocketManager] = None):
        # We'll initialize these lazily if not provided
        self.pipeline_service = pipeline_service
        self.websocket_manager = websocket_manager
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.result_ttl = settings.RESULT_TTL
        
    async def initialize(self):
        """Initialize the workflow manager"""
        logger.info("Initializing workflow manager...")
        
        # Initialize the pipeline service if it exists
        if self.pipeline_service is None:
            self.pipeline_service = PipelineService()
        await self.pipeline_service.initialize()
        
        # Initialize the WebSocket manager if it exists
        if self.websocket_manager is None:
            self.websocket_manager = WebSocketManager()
        await self.websocket_manager.initialize()
        
        # Load any pending jobs from Redis
        await self._load_pending_jobs()
        
        logger.info("Workflow manager initialized successfully")
        
    async def _load_pending_jobs(self):
        """Load pending jobs from Redis"""
        try:
            # Get all job keys
            job_keys = await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.keys,
                "job:*"
            )
            
            for key in job_keys:
                job_id = key.decode('utf-8').split(':')[1]
                
                # Get job data
                job_data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.hgetall,
                    f"job:{job_id}"
                )
                
                # Convert bytes to string
                job_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in job_data.items()}
                
                # Check if job is still active
                if job_data.get('status') in [JobStatus.QUEUED.value, JobStatus.PROCESSING.value]:
                    # Add to active jobs
                    self.active_jobs[job_id] = {
                        'status': JobStatus(job_data.get('status')),
                        'created_at': float(job_data.get('created_at', time.time())),
                        'request': json.loads(job_data.get('request', '{}')),
                        'result': None
                    }
                    
                    logger.info(f"Loaded pending job {job_id}")
        except Exception as e:
            logger.error(f"Failed to load pending jobs: {str(e)}")
    
    async def create_job(self, request_data: Dict[str, Any]) -> str:
        """Create a new job and return job ID"""
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create processing request
        request = ProcessingRequest(
            job_id=job_id,
            bounds=request_data.get('bounds'),
            zoom_level=request_data.get('zoom_level', 18),
            enable_rt=request_data.get('enable_rt', True),
            enable_rr=request_data.get('enable_rr', True),
            enable_fer=request_data.get('enable_fer', True),
            enable_rl_fusion=request_data.get('enable_rl_fusion', True)
        )
        
        # Store job in Redis
        await self._store_job(job_id, request)
        
        # Add to active jobs
        self.active_jobs[job_id] = {
            'status': JobStatus.QUEUED,
            'created_at': time.time(),
            'request': request_data,
            'result': None
        }
        
        # Schedule job processing
        asyncio.create_task(self._process_job(job_id, request))
        
        return job_id
    
    async def _store_job(self, job_id: str, request: ProcessingRequest):
        """Store job in Redis"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._store_job_sync(job_id, request)
        )
        
    def _store_job_sync(self, job_id: str, request: ProcessingRequest):
        """Synchronous helper for storing job in Redis"""
        self.redis_client.hset(
            f"job:{job_id}",
            mapping={
                'status': JobStatus.QUEUED.value,
                'created_at': str(time.time()),
                'request': request.model_dump_json()
            }
        )
        # Set expiration for job data
        self.redis_client.expire(f"job:{job_id}", self.result_ttl)
        
    async def _process_job(self, job_id: str, request: ProcessingRequest):
        """Process a job"""
        try:
            # Update job status
            await self._update_job_status(job_id, JobStatus.PROCESSING)
            
            # Process the request
            result = await self.pipeline_service.process_request(
                request,
                progress_callback=self._progress_callback
            )
            
            # Store result
            await self._store_result(job_id, result)
            
            # Update job status
            await self._update_job_status(job_id, JobStatus.COMPLETED)
            
            # Notify WebSocket clients
            await self.websocket_manager.broadcast_job_update(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                stage=ProcessingStage.COMPLETED,
                message="Processing completed successfully"
            )
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            
            # Update job status
            await self._update_job_status(job_id, JobStatus.FAILED)
            
            # Notify WebSocket clients
            await self.websocket_manager.broadcast_job_update(
                job_id=job_id,
                status=JobStatus.FAILED,
                progress=0.0,
                stage=ProcessingStage.FAILED,
                message=f"Processing failed: {str(e)}"
            )
            
        finally:
            # Remove from active jobs after processing
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _update_job_status(self, job_id: str, status: JobStatus):
        """Update job status in Redis"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis_client.hset,
            f"job:{job_id}",
            "status",
            status.value
        )
        
        # Update active jobs
        if job_id in self.active_jobs:
            self.active_jobs[job_id]['status'] = status
    
    async def _store_result(self, job_id: str, result: Dict[str, Any]):
        """Store job result in Redis"""
        result_key = f"result:{job_id}"
        
        # Serialize result
        result_json = json.dumps(result)
        
        # Store in Redis
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._store_result_sync(result_key, result_json)
        )
        
        # Update active jobs
        if job_id in self.active_jobs:
            self.active_jobs[job_id]['result'] = result
    
    def _store_result_sync(self, result_key: str, result_json: str):
        """Synchronous helper for storing result in Redis"""
        self.redis_client.set(result_key, result_json)
        self.redis_client.expire(result_key, self.result_ttl)
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        # Check active jobs first
        if job_id in self.active_jobs:
            return {
                'job_id': job_id,
                'status': self.active_jobs[job_id]['status'].value,
                'created_at': self.active_jobs[job_id]['created_at']
            }
        
        # Check Redis
        job_data = await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis_client.hgetall,
            f"job:{job_id}"
        )
        
        if not job_data:
            return {
                'job_id': job_id,
                'status': JobStatus.NOT_FOUND.value
            }
        
        # Convert bytes to string
        job_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in job_data.items()}
        
        return {
            'job_id': job_id,
            'status': job_data.get('status', JobStatus.NOT_FOUND.value),
            'created_at': float(job_data.get('created_at', 0))
        }
    
    async def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get job result"""
        # Check active jobs first
        if job_id in self.active_jobs and self.active_jobs[job_id].get('result') is not None:
            return self.active_jobs[job_id]['result']
        
        # Check Redis
        result_json = await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis_client.get,
            f"result:{job_id}"
        )
        
        if not result_json:
            return None
        
        return json.loads(result_json.decode('utf-8'))
    
    async def _progress_callback(self, job_id: str, stage: ProcessingStage, progress: float, message: str):
        """Callback for processing progress"""
        # Notify WebSocket clients
        await self.websocket_manager.broadcast_job_update(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=progress,
            stage=stage,
            message=message
        )
            
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up workflow manager resources...")
        # Close Redis connection - if using aioredis you might have different cleanup
        try:
            # For standard redis-py
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")
        
        # Additional cleanup if needed
        logger.info("Workflow manager cleanup completed")