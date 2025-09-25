"""
WebSocket Manager for real-time updates via Socket.IO
"""

import logging
from typing import Dict, Any, Optional
from app.models.schemas import JobStatus, ProcessingStage
from app.core.websocket import sio_app, connected_clients

logger = logging.getLogger(__name__)

class WebSocketManager:
    """WebSocket Manager using Socket.IO for real-time updates"""
    
    def __init__(self):
        """Initialize WebSocket manager"""
        pass
    
    async def initialize(self):
        """Initialize WebSocket manager"""
        logger.info("WebSocket manager initialized")
    
    async def broadcast_job_update(
        self, 
        job_id: str, 
        status: JobStatus, 
        progress: float,
        stage: ProcessingStage,
        message: str
    ):
        """
        Broadcast job update to all subscribed clients
        
        Args:
            job_id: Job ID
            status: Job status
            progress: Progress (0-1)
            stage: Processing stage
            message: Progress message
        """
        data = {
            "job_id": job_id,
            "status": status.value,
            "progress": progress,
            "stage": stage.value,
            "message": message
        }
        
        # Broadcast to all clients subscribed to this job
        for sid, client in connected_clients.items():
            if job_id in client.get("jobs", []):
                await sio_app.emit("job_progress", data, to=sid)
                logger.debug(f"Progress update sent to client {sid} for job {job_id}")
        
        logger.debug(f"Job {job_id}: {stage.value} - {progress:.2f} - {message}")
    
    async def broadcast_system_message(self, message: str, level: str = "info"):
        """
        Broadcast system message to all clients
        
        Args:
            message: Message to broadcast
            level: Message level (info, warning, error)
        """
        data = {
            "type": "system",
            "message": message,
            "level": level
        }
        
        await sio_app.emit("system", data)