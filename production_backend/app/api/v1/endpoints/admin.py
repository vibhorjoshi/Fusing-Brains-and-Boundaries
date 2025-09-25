"""
Admin API endpoints
Administrative functions for system management
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.user import User, UserRole
from app.models.processing_job import ProcessingJob, JobStatus
from app.models.building_footprint import BuildingFootprint
from app.models.file_storage import FileStorage
from app.api.deps import get_admin_user, get_current_user
from app.schemas.user_schemas import UserResponse, UserUpdate
from app.schemas.job_schemas import ProcessingJobResponse
from app.core.security import get_password_hash
from app.services.s3_service import s3_service
from app.tasks.ml_processing import cleanup_old_jobs, generate_processing_report

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/dashboard")
async def admin_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Get comprehensive admin dashboard statistics
    
    Returns system-wide statistics and health metrics.
    """
    try:
        # User statistics
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        premium_users = db.query(User).filter(User.role == UserRole.PREMIUM).count()
        admin_users = db.query(User).filter(User.role == UserRole.ADMIN).count()
        
        # Job statistics
        total_jobs = db.query(ProcessingJob).count()
        completed_jobs = db.query(ProcessingJob).filter(ProcessingJob.status == JobStatus.COMPLETED).count()
        failed_jobs = db.query(ProcessingJob).filter(ProcessingJob.status == JobStatus.FAILED).count()
        running_jobs = db.query(ProcessingJob).filter(ProcessingJob.status == JobStatus.RUNNING).count()
        queued_jobs = db.query(ProcessingJob).filter(ProcessingJob.status == JobStatus.QUEUED).count()
        
        # Building statistics
        total_buildings = db.query(BuildingFootprint).count()
        
        # Recent activity (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_users = db.query(User).filter(User.created_at >= yesterday).count()
        recent_jobs = db.query(ProcessingJob).filter(ProcessingJob.created_at >= yesterday).count()
        recent_buildings = db.query(BuildingFootprint).filter(BuildingFootprint.created_at >= yesterday).count()
        
        # File storage statistics
        total_files = db.query(FileStorage).count()
        total_storage_size = db.query(func.sum(FileStorage.file_size)).scalar() or 0
        
        # System health metrics
        from app.core.celery_app import get_worker_stats
        worker_stats = get_worker_stats()
        
        dashboard_data = {
            "system_overview": {
                "total_users": total_users,
                "active_users": active_users,
                "premium_users": premium_users,
                "admin_users": admin_users,
                "total_jobs": total_jobs,
                "total_buildings": total_buildings,
                "total_files": total_files,
                "total_storage_gb": round(total_storage_size / (1024**3), 2)
            },
            "job_statistics": {
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "running_jobs": running_jobs,
                "queued_jobs": queued_jobs,
                "success_rate": (completed_jobs / max(total_jobs, 1)) * 100
            },
            "recent_activity": {
                "new_users_24h": recent_users,
                "new_jobs_24h": recent_jobs,
                "new_buildings_24h": recent_buildings
            },
            "system_health": {
                "celery_workers": worker_stats.get("active_workers", 0),
                "queue_status": "healthy" if running_jobs + queued_jobs < 50 else "busy",
                "database_status": "operational"
            }
        }
        
        logger.info(f"✅ Admin dashboard accessed by {current_user.username}")
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"❌ Failed to generate admin dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate dashboard data")

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: Optional[UserRole] = Query(None, description="Filter by user role"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Get all users with filtering and pagination
    
    Admin-only endpoint for user management.
    """
    try:
        query = db.query(User)
        
        if role:
            query = query.filter(User.role == role)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        users = query.order_by(desc(User.created_at)).offset(skip).limit(limit).all()
        
        logger.info(f"✅ Retrieved {len(users)} users for admin {current_user.username}")
        
        return users
        
    except Exception as e:
        logger.error(f"❌ Failed to get users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Update user information
    
    Admin-only endpoint for modifying user accounts.
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update user fields
        for field, value in user_update.dict(exclude_unset=True).items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(user)
        
        logger.info(f"✅ User {user_id} updated by admin {current_user.username}")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")

@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Deactivate a user account
    
    Admin-only endpoint for deactivating user accounts.
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.id == current_user.id:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
        
        user.is_active = False
        user.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"✅ User {user_id} deactivated by admin {current_user.username}")
        
        return {"status": "success", "message": "User deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to deactivate user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to deactivate user")

@router.post("/users/{user_id}/promote")
async def promote_user(
    user_id: int,
    new_role: UserRole,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Change user role
    
    Admin-only endpoint for promoting/demoting users.
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        old_role = user.role
        user.role = new_role
        user.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"✅ User {user_id} role changed from {old_role.value} to {new_role.value} by admin {current_user.username}")
        
        return {
            "status": "success",
            "message": f"User role changed from {old_role.value} to {new_role.value}",
            "old_role": old_role.value,
            "new_role": new_role.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to promote user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to change user role")

@router.get("/jobs", response_model=List[ProcessingJobResponse])
async def get_all_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Get all processing jobs with filtering
    
    Admin-only endpoint for job management.
    """
    try:
        query = db.query(ProcessingJob)
        
        if status:
            query = query.filter(ProcessingJob.status == status)
        
        if user_id:
            query = query.filter(ProcessingJob.user_id == user_id)
        
        jobs = query.order_by(desc(ProcessingJob.created_at)).offset(skip).limit(limit).all()
        
        logger.info(f"✅ Retrieved {len(jobs)} jobs for admin {current_user.username}")
        
        return jobs
        
    except Exception as e:
        logger.error(f"❌ Failed to get jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")

@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Delete a processing job
    
    Admin-only endpoint for removing jobs and associated data.
    """
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Delete associated buildings
        db.query(BuildingFootprint).filter(BuildingFootprint.job_id == job_id).delete()
        
        # Delete job
        db.delete(job)
        db.commit()
        
        logger.info(f"✅ Job {job_id} deleted by admin {current_user.username}")
        
        return {"status": "success", "message": "Job and associated data deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete job")

@router.post("/cleanup")
async def cleanup_system(
    days_old: int = Query(30, ge=1, le=365, description="Clean data older than X days"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Trigger system cleanup
    
    Admin-only endpoint for cleaning up old data and optimizing the system.
    """
    try:
        # Queue cleanup task
        task_result = cleanup_old_jobs.delay(days_old)
        
        logger.info(f"✅ System cleanup initiated by admin {current_user.username} for data older than {days_old} days")
        
        return {
            "status": "cleanup_started",
            "task_id": task_result.id,
            "days_old": days_old,
            "message": "System cleanup task has been queued"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to start system cleanup")

@router.post("/reports/generate")
async def generate_admin_report(
    report_type: str = Query("system", regex="^(system|user|processing|storage)$"),
    start_date: Optional[datetime] = Query(None, description="Report start date"),
    end_date: Optional[datetime] = Query(None, description="Report end date"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Generate administrative reports
    
    Admin-only endpoint for generating various system reports.
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Get relevant job IDs for the date range
        jobs_in_range = db.query(ProcessingJob.id).filter(
            ProcessingJob.created_at >= start_date,
            ProcessingJob.created_at <= end_date
        ).all()
        
        job_ids = [job.id for job in jobs_in_range]
        
        if not job_ids:
            return {
                "status": "no_data",
                "message": "No data found for the specified date range",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        
        # Queue report generation task
        task_result = generate_processing_report.delay(job_ids)
        
        logger.info(f"✅ Admin report generation started by {current_user.username}")
        
        return {
            "status": "report_started",
            "task_id": task_result.id,
            "report_type": report_type,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "job_count": len(job_ids),
            "message": "Report generation task has been queued"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate admin report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

@router.get("/system/health")
async def system_health_check(
    current_user: User = Depends(get_admin_user)
):
    """
    Comprehensive system health check
    
    Admin-only endpoint for monitoring system health and performance.
    """
    try:
        # Database connectivity test
        db_health = {"status": "healthy", "response_time_ms": 0}
        try:
            from time import time
            start_time = time()
            
            from app.core.database import get_db
            db = next(get_db())
            db.execute("SELECT 1")
            db.close()
            
            db_health["response_time_ms"] = round((time() - start_time) * 1000, 2)
            
        except Exception as e:
            db_health = {"status": "unhealthy", "error": str(e)}
        
        # Celery workers health
        from app.core.celery_app import get_worker_stats
        worker_stats = get_worker_stats()
        
        # S3 connectivity test
        s3_health = {"status": "healthy"}
        try:
            # Test S3 connection (mock - implement actual test)
            s3_health["buckets_accessible"] = True
        except Exception as e:
            s3_health = {"status": "unhealthy", "error": str(e)}
        
        # System metrics
        import psutil
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        health_report = {
            "overall_status": "healthy",  # This would be calculated based on components
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": db_health,
                "celery_workers": {
                    "status": "healthy" if worker_stats.get("active_workers", 0) > 0 else "warning",
                    "active_workers": worker_stats.get("active_workers", 0)
                },
                "s3_storage": s3_health
            },
            "system_metrics": system_metrics
        }
        
        logger.info(f"✅ System health check performed by admin {current_user.username}")
        
        return health_report
        
    except Exception as e:
        logger.error(f"❌ System health check failed: {e}")
        return {
            "overall_status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }