"""
WebSocket handling via Socket.IO integration
"""

import socketio
import logging
from typing import Dict, Any, List, Optional

# Create Socket.IO server
sio_app = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*"
)

# Set up logging
logger = logging.getLogger(__name__)

# Connected clients
connected_clients: Dict[str, Any] = {}

@sio_app.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    logger.info(f"Client connected: {sid}")
    connected_clients[sid] = {
        "sid": sid,
        "connected_at": socketio.time.time(),
        "jobs": []
    }
    await sio_app.emit("connection_status", {"status": "connected"}, to=sid)

@sio_app.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")
    if sid in connected_clients:
        del connected_clients[sid]

@sio_app.event
async def join_job(sid, data):
    """Subscribe client to job updates"""
    job_id = data.get("job_id")
    if not job_id:
        await sio_app.emit("error", {"message": "Job ID is required"}, to=sid)
        return
    
    if sid in connected_clients:
        connected_clients[sid]["jobs"].append(job_id)
        await sio_app.emit("job_subscribed", {"job_id": job_id}, to=sid)
        logger.info(f"Client {sid} subscribed to job {job_id}")

@sio_app.event
async def leave_job(sid, data):
    """Unsubscribe client from job updates"""
    job_id = data.get("job_id")
    if not job_id:
        await sio_app.emit("error", {"message": "Job ID is required"}, to=sid)
        return
    
    if sid in connected_clients and job_id in connected_clients[sid]["jobs"]:
        connected_clients[sid]["jobs"].remove(job_id)
        await sio_app.emit("job_unsubscribed", {"job_id": job_id}, to=sid)
        logger.info(f"Client {sid} unsubscribed from job {job_id}")

async def broadcast_job_update(job_id: str, update_type: str, data: Dict[str, Any]):
    """Broadcast job update to all subscribed clients"""
    for sid, client in connected_clients.items():
        if job_id in client["jobs"]:
            await sio_app.emit(
                update_type, 
                {"job_id": job_id, **data}, 
                to=sid
            )
    
    logger.info(f"Broadcast {update_type} for job {job_id} sent")