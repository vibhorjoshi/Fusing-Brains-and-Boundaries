"""
Redis Integration Module
Handles Redis connections, caching, and session management
"""

import json
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError

logger = logging.getLogger(__name__)

class RedisManager:
    """Redis connection and operations manager"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.client: Optional[redis.Redis] = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Establish Redis connection"""
        try:
            self.client = redis.from_url(
                self.redis_url, 
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            self.connected = True
            logger.info("✅ Redis connection established")
            return True
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("🔄 Redis connection closed")
    
    async def ping(self) -> bool:
        """Check Redis connectivity"""
        if not self.client or not self.connected:
            return False
        
        try:
            return await self.client.ping()
        except Exception:
            self.connected = False
            return False
    
    # Session Management
    async def set_session(self, session_id: str, data: Dict[str, Any], expiry: int = 1800):
        """Store user session data"""
        if not self.connected:
            return False
        
        try:
            session_data = {
                "data": json.dumps(data),
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(seconds=expiry)).isoformat()
            }
            
            return await self.client.setex(
                f"session:{session_id}", 
                expiry, 
                json.dumps(session_data)
            )
        except Exception as e:
            logger.error(f"Failed to set session {session_id}: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user session data"""
        if not self.connected:
            return None
        
        try:
            session_data = await self.client.get(f"session:{session_id}")
            if session_data:
                parsed_data = json.loads(session_data)
                return json.loads(parsed_data["data"])
            return None
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Remove user session"""
        if not self.connected:
            return False
        
        try:
            result = await self.client.delete(f"session:{session_id}")
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    # Caching Operations
    async def set_cache(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """Set cache with optional expiry"""
        if not self.connected:
            return False
        
        try:
            cache_value = json.dumps(value) if not isinstance(value, str) else value
            
            if expiry:
                return await self.client.setex(f"cache:{key}", expiry, cache_value)
            else:
                return await self.client.set(f"cache:{key}", cache_value)
        except Exception as e:
            logger.error(f"Failed to set cache {key}: {e}")
            return False
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if not self.connected:
            return None
        
        try:
            value = await self.client.get(f"cache:{key}")
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Failed to get cache {key}: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cached value"""
        if not self.connected:
            return False
        
        try:
            result = await self.client.delete(f"cache:{key}")
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete cache {key}: {e}")
            return False
    
    # User Data Storage
    async def store_user_data(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Store user-related data"""
        if not self.connected:
            return False
        
        try:
            return await self.client.hset(f"user:{user_id}", mapping=data)
        except Exception as e:
            logger.error(f"Failed to store user data {user_id}: {e}")
            return False
    
    async def get_user_data(self, user_id: str, field: Optional[str] = None) -> Optional[Union[Dict, str]]:
        """Get user data"""
        if not self.connected:
            return None
        
        try:
            if field:
                return await self.client.hget(f"user:{user_id}", field)
            else:
                return await self.client.hgetall(f"user:{user_id}")
        except Exception as e:
            logger.error(f"Failed to get user data {user_id}: {e}")
            return None
    
    # Job/Task Storage
    async def store_job_data(self, job_id: str, data: Dict[str, Any]) -> bool:
        """Store processing job data"""
        if not self.connected:
            return False
        
        try:
            # Convert datetime objects to strings
            serialized_data = {}
            for key, value in data.items():
                if isinstance(value, datetime):
                    serialized_data[key] = value.isoformat()
                elif isinstance(value, dict) or isinstance(value, list):
                    serialized_data[key] = json.dumps(value)
                else:
                    serialized_data[key] = str(value)
            
            return await self.client.hset(f"job:{job_id}", mapping=serialized_data)
        except Exception as e:
            logger.error(f"Failed to store job data {job_id}: {e}")
            return False
    
    async def get_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get processing job data"""
        if not self.connected:
            return None
        
        try:
            data = await self.client.hgetall(f"job:{job_id}")
            if data:
                # Parse JSON fields back
                parsed_data = {}
                for key, value in data.items():
                    if key in ["results", "request_data"] and value:
                        try:
                            parsed_data[key] = json.loads(value)
                        except json.JSONDecodeError:
                            parsed_data[key] = value
                    else:
                        parsed_data[key] = value
                return parsed_data
            return None
        except Exception as e:
            logger.error(f"Failed to get job data {job_id}: {e}")
            return None
    
    async def update_job_progress(self, job_id: str, progress: float, status: str, stage: Optional[str] = None) -> bool:
        """Update job progress"""
        if not self.connected:
            return False
        
        try:
            update_data = {
                "progress": str(progress),
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            if stage:
                update_data["current_stage"] = stage
            
            return await self.client.hset(f"job:{job_id}", mapping=update_data)
        except Exception as e:
            logger.error(f"Failed to update job progress {job_id}: {e}")
            return False
    
    # Rate Limiting
    async def check_rate_limit(self, key: str, limit: int, window: int) -> Dict[str, Union[bool, int]]:
        """Check rate limit for a given key"""
        if not self.connected:
            return {"allowed": True, "remaining": limit, "reset_time": 0}
        
        try:
            current_count = await self.client.get(f"rate_limit:{key}")
            
            if current_count is None:
                # First request in window
                await self.client.setex(f"rate_limit:{key}", window, 1)
                return {
                    "allowed": True,
                    "remaining": limit - 1,
                    "reset_time": window
                }
            
            count = int(current_count)
            
            if count >= limit:
                ttl = await self.client.ttl(f"rate_limit:{key}")
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_time": ttl if ttl > 0 else window
                }
            
            # Increment counter
            new_count = await self.client.incr(f"rate_limit:{key}")
            ttl = await self.client.ttl(f"rate_limit:{key}")
            
            return {
                "allowed": True,
                "remaining": max(0, limit - new_count),
                "reset_time": ttl if ttl > 0 else window
            }
            
        except Exception as e:
            logger.error(f"Failed to check rate limit for {key}: {e}")
            return {"allowed": True, "remaining": limit, "reset_time": 0}
    
    # Statistics and Analytics
    async def increment_metric(self, metric_name: str, amount: int = 1) -> bool:
        """Increment a metric counter"""
        if not self.connected:
            return False
        
        try:
            await self.client.incr(f"metric:{metric_name}", amount)
            return True
        except Exception as e:
            logger.error(f"Failed to increment metric {metric_name}: {e}")
            return False
    
    async def get_metric(self, metric_name: str) -> int:
        """Get metric value"""
        if not self.connected:
            return 0
        
        try:
            value = await self.client.get(f"metric:{metric_name}")
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Failed to get metric {metric_name}: {e}")
            return 0
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get Redis system information"""
        if not self.connected or not self.client:
            return {"connected": False}
        
        try:
            info = await self.client.info()
            return {
                "connected": True,
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {"connected": False, "error": str(e)}
    
    # Cleanup Operations
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired session data"""
        if not self.connected:
            return 0
        
        try:
            # Get all session keys
            session_keys = await self.client.keys("session:*")
            expired_count = 0
            
            for key in session_keys:
                ttl = await self.client.ttl(key)
                if ttl == -1:  # No expiry set, check manually
                    session_data = await self.client.get(key)
                    if session_data:
                        try:
                            data = json.loads(session_data)
                            expires_at = datetime.fromisoformat(data.get("expires_at", ""))
                            if expires_at < datetime.now():
                                await self.client.delete(key)
                                expired_count += 1
                        except (json.JSONDecodeError, ValueError):
                            # Invalid data, delete
                            await self.client.delete(key)
                            expired_count += 1
            
            return expired_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0


# Global Redis manager instance
redis_manager: Optional[RedisManager] = None

async def get_redis() -> RedisManager:
    """Get Redis manager instance"""
    global redis_manager
    if not redis_manager:
        from app.core.config import settings
        redis_manager = RedisManager(settings.REDIS_URL)
        await redis_manager.connect()
    return redis_manager

async def init_redis(redis_url: str) -> bool:
    """Initialize Redis connection"""
    global redis_manager
    redis_manager = RedisManager(redis_url)
    return await redis_manager.connect()

async def close_redis():
    """Close Redis connection"""
    global redis_manager
    if redis_manager:
        await redis_manager.disconnect()
        redis_manager = None