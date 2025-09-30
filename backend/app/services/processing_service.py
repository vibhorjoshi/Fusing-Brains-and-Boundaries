"""
Processing Service for GeoAI Research Backend
Handles processing job data access and management
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from ..models.processing_models import ProcessingJob, ProcessingStatus
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingService:
    """Service for managing processing jobs"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.jobs_file = os.path.join(data_dir, "processing_jobs.json")
        self._init_storage_files()
    
    def _init_storage_files(self):
        """Initialize storage files if they don't exist"""
        if not os.path.exists(self.jobs_file):
            with open(self.jobs_file, 'w') as f:
                json.dump({}, f)
    
    async def create_job(self, job: ProcessingJob) -> ProcessingJob:
        """Create a new processing job"""
        try:
            jobs_data = self._load_json(self.jobs_file)
            
            # Convert datetime objects to strings for JSON storage
            job_data = job.dict()
            if job_data.get("created_at"):
                job_data["created_at"] = job_data["created_at"].isoformat()
            if job_data.get("started_at"):
                job_data["started_at"] = job_data["started_at"].isoformat()
            if job_data.get("completed_at"):
                job_data["completed_at"] = job_data["completed_at"].isoformat()
            
            jobs_data[job.job_id] = job_data
            self._save_json(self.jobs_file, jobs_data)
            
            logger.info(f"Processing job created: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error creating processing job: {str(e)}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get processing job by ID"""
        try:
            jobs_data = self._load_json(self.jobs_file)
            job_data = jobs_data.get(job_id)
            
            if job_data:
                # Convert ISO strings back to datetime objects
                if job_data.get("created_at"):
                    job_data["created_at"] = datetime.fromisoformat(job_data["created_at"])
                if job_data.get("started_at"):
                    job_data["started_at"] = datetime.fromisoformat(job_data["started_at"])
                if job_data.get("completed_at"):
                    job_data["completed_at"] = datetime.fromisoformat(job_data["completed_at"])
                
                return ProcessingJob(**job_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting processing job: {str(e)}")
            return None
    
    async def update_job_progress(self, job_id: str, progress: float) -> bool:
        """Update job progress"""
        try:
            jobs_data = self._load_json(self.jobs_file)
            
            if job_id in jobs_data:
                jobs_data[job_id]["progress"] = progress
                self._save_json(self.jobs_file, jobs_data)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating job progress: {str(e)}")
            return False
    
    async def complete_job(self, job_id: str, results: Dict) -> bool:
        """Complete a processing job"""
        try:
            jobs_data = self._load_json(self.jobs_file)
            
            if job_id in jobs_data:
                jobs_data[job_id]["status"] = ProcessingStatus.COMPLETED.value
                jobs_data[job_id]["results"] = results
                jobs_data[job_id]["completed_at"] = datetime.now().isoformat()
                jobs_data[job_id]["progress"] = 100.0
                
                self._save_json(self.jobs_file, jobs_data)
                logger.info(f"Processing job completed: {job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error completing job: {str(e)}")
            return False
    
    async def fail_job(self, job_id: str, error_message: str) -> bool:
        """Mark job as failed"""
        try:
            jobs_data = self._load_json(self.jobs_file)
            
            if job_id in jobs_data:
                jobs_data[job_id]["status"] = ProcessingStatus.FAILED.value
                jobs_data[job_id]["error_message"] = error_message
                jobs_data[job_id]["completed_at"] = datetime.now().isoformat()
                
                self._save_json(self.jobs_file, jobs_data)
                logger.error(f"Processing job failed: {job_id} - {error_message}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error failing job: {str(e)}")
            return False
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job"""
        try:
            jobs_data = self._load_json(self.jobs_file)
            
            if job_id in jobs_data:
                jobs_data[job_id]["status"] = ProcessingStatus.CANCELLED.value
                jobs_data[job_id]["completed_at"] = datetime.now().isoformat()
                
                self._save_json(self.jobs_file, jobs_data)
                logger.info(f"Processing job cancelled: {job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            return False
    
    async def get_user_jobs(self, user_id: int, limit: int = 50) -> List[ProcessingJob]:
        """Get jobs for a specific user"""
        try:
            jobs_data = self._load_json(self.jobs_file)
            user_jobs = []
            
            for job_id, job_data in jobs_data.items():
                if job_data.get("user_id") == user_id:
                    # Convert ISO strings back to datetime objects
                    if job_data.get("created_at"):
                        job_data["created_at"] = datetime.fromisoformat(job_data["created_at"])
                    if job_data.get("started_at"):
                        job_data["started_at"] = datetime.fromisoformat(job_data["started_at"])
                    if job_data.get("completed_at"):
                        job_data["completed_at"] = datetime.fromisoformat(job_data["completed_at"])
                    
                    user_jobs.append(ProcessingJob(**job_data))
            
            # Sort by creation date (newest first) and limit results
            user_jobs.sort(key=lambda x: x.created_at, reverse=True)
            return user_jobs[:limit]
            
        except Exception as e:
            logger.error(f"Error getting user jobs: {str(e)}")
            return []
    
    async def get_active_jobs(self) -> List[ProcessingJob]:
        """Get all active (processing) jobs"""
        try:
            jobs_data = self._load_json(self.jobs_file)
            active_jobs = []
            
            for job_id, job_data in jobs_data.items():
                if job_data.get("status") == ProcessingStatus.PROCESSING.value:
                    # Convert ISO strings back to datetime objects
                    if job_data.get("created_at"):
                        job_data["created_at"] = datetime.fromisoformat(job_data["created_at"])
                    if job_data.get("started_at"):
                        job_data["started_at"] = datetime.fromisoformat(job_data["started_at"])
                    if job_data.get("completed_at"):
                        job_data["completed_at"] = datetime.fromisoformat(job_data["completed_at"])
                    
                    active_jobs.append(ProcessingJob(**job_data))
            
            return active_jobs
            
        except Exception as e:
            logger.error(f"Error getting active jobs: {str(e)}")
            return []
    
    def _load_json(self, file_path: str) -> Dict:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_json(self, file_path: str, data: Dict) -> None:
        """Save JSON data to file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)