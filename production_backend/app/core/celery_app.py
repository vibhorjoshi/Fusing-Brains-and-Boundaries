"""
Celery configuration for background task processing
Production-ready distributed task queue for ML operations
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import logging
from datetime import datetime
import os

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery instance
celery_app = Celery(
    "building_footprint_ai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.ml_processing",
        "app.tasks.file_processing", 
        "app.tasks.data_processing",
        "app.tasks.batch_processing"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    
    # Task routing
    task_routes={
        "app.tasks.ml_processing.*": {"queue": "ml_processing"},
        "app.tasks.file_processing.*": {"queue": "file_processing"},
        "app.tasks.data_processing.*": {"queue": "data_processing"},
        "app.tasks.batch_processing.*": {"queue": "batch_processing"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_default_exchange="default",
    task_default_exchange_type="direct",
    task_default_routing_key="default",
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Important for ML tasks
    worker_max_tasks_per_child=10,  # Prevent memory leaks
    worker_disable_rate_limits=False,
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
    
    # Task execution settings
    task_acks_late=True,  # Acknowledge only after task completion
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hours max per task
    task_soft_time_limit=6600,  # 1h 50m soft limit
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup_old_files": {
            "task": "app.tasks.file_processing.cleanup_old_files",
            "schedule": 3600.0,  # Every hour
        },
        "update_job_statistics": {
            "task": "app.tasks.data_processing.update_job_statistics",
            "schedule": 1800.0,  # Every 30 minutes
        },
        "health_check_services": {
            "task": "app.tasks.data_processing.health_check_services",
            "schedule": 900.0,  # Every 15 minutes
        },
    },
    
    # Redis settings for better reliability
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
)

# Task monitoring signals
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Log task start"""
    logger.info(f"🚀 Task started: {task.name} (ID: {task_id})")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Log task completion"""
    logger.info(f"✅ Task completed: {task.name} (ID: {task_id}) - State: {state}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Log task failure"""
    logger.error(f"❌ Task failed: {sender.name} (ID: {task_id}) - Exception: {exception}")

# Health check task
@celery_app.task(bind=True)
def health_check(self):
    """Basic health check task"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "worker_id": self.request.hostname,
        "task_id": self.request.id
    }

# Get task info
def get_task_info(task_id: str):
    """Get information about a specific task"""
    try:
        result = celery_app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
            "ready": result.ready(),
            "successful": result.successful(),
            "failed": result.failed(),
            "traceback": result.traceback
        }
    except Exception as e:
        return {"error": str(e)}

# Get worker statistics
def get_worker_stats():
    """Get statistics about active workers"""
    try:
        inspect = celery_app.control.inspect()
        
        # Get active tasks
        active_tasks = inspect.active()
        
        # Get registered tasks
        registered_tasks = inspect.registered()
        
        # Get worker statistics
        stats = inspect.stats()
        
        return {
            "active_tasks": active_tasks,
            "registered_tasks": registered_tasks,
            "worker_stats": stats,
            "queue_lengths": get_queue_lengths()
        }
    except Exception as e:
        return {"error": str(e)}

def get_queue_lengths():
    """Get current queue lengths"""
    try:
        inspect = celery_app.control.inspect()
        reserved = inspect.reserved()
        
        queue_lengths = {}
        for worker, tasks in (reserved or {}).items():
            for task in tasks:
                queue = task.get("delivery_info", {}).get("routing_key", "default")
                queue_lengths[queue] = queue_lengths.get(queue, 0) + 1
        
        return queue_lengths
    except Exception as e:
        logger.error(f"Failed to get queue lengths: {e}")
        return {}

# Purge all queues (use with caution!)
def purge_all_queues():
    """Purge all task queues"""
    try:
        celery_app.control.purge()
        logger.warning("⚠️ All Celery queues purged")
        return True
    except Exception as e:
        logger.error(f"Failed to purge queues: {e}")
        return False

# Shutdown workers gracefully
def shutdown_workers():
    """Shutdown all workers gracefully"""
    try:
        celery_app.control.shutdown()
        logger.info("🔄 Celery workers shutdown initiated")
        return True
    except Exception as e:
        logger.error(f"Failed to shutdown workers: {e}")
        return False