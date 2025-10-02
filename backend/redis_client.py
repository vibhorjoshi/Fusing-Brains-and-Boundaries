from redis import Redis
import aioredis
import os
import json
from typing import Any, Dict, List, Optional, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Redis connection parameters
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Redis key prefixes
JOB_PREFIX = "job:"
RESULT_PREFIX = "result:"
STATS_PREFIX = "stats:"
CACHE_PREFIX = "cache:"

# Connect to Redis
def get_redis_connection() -> Redis:
    """Get synchronous Redis connection"""
    try:
        redis = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
        )
        # Test connection
        redis.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return redis
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

async def get_async_redis_connection() -> aioredis.Redis:
    """Get asynchronous Redis connection"""
    try:
        redis = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
            password=REDIS_PASSWORD,
            decode_responses=True,
        )
        # Test connection
        await redis.ping()
        logger.info(f"Connected to async Redis at {REDIS_HOST}:{REDIS_PORT}")
        return redis
    except Exception as e:
        logger.error(f"Failed to connect to async Redis: {e}")
        raise

# Job storage methods
async def store_job(job_id: str, job_data: Dict[str, Any]) -> bool:
    """Store job data in Redis"""
    redis = await get_async_redis_connection()
    try:
        # Convert data to JSON
        job_json = json.dumps(job_data)
        
        # Store in Redis
        await redis.set(f"{JOB_PREFIX}{job_id}", job_json)
        
        # Add to jobs list
        await redis.lpush("jobs", job_id)
        
        return True
    except Exception as e:
        logger.error(f"Error storing job {job_id}: {e}")
        return False
    finally:
        await redis.close()

async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job data from Redis"""
    redis = await get_async_redis_connection()
    try:
        job_json = await redis.get(f"{JOB_PREFIX}{job_id}")
        if not job_json:
            return None
        
        return json.loads(job_json)
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}")
        return None
    finally:
        await redis.close()

async def update_job(job_id: str, updates: Dict[str, Any]) -> bool:
    """Update job data in Redis"""
    redis = await get_async_redis_connection()
    try:
        # Get existing job
        job_json = await redis.get(f"{JOB_PREFIX}{job_id}")
        if not job_json:
            return False
        
        job_data = json.loads(job_json)
        
        # Update with new values
        job_data.update(updates)
        
        # Store updated job
        await redis.set(f"{JOB_PREFIX}{job_id}", json.dumps(job_data))
        
        return True
    except Exception as e:
        logger.error(f"Error updating job {job_id}: {e}")
        return False
    finally:
        await redis.close()

async def list_jobs(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """List jobs from Redis"""
    redis = await get_async_redis_connection()
    try:
        # Get job IDs
        job_ids = await redis.lrange("jobs", offset, offset + limit - 1)
        
        # Get job data for each ID
        jobs = []
        for job_id in job_ids:
            job_json = await redis.get(f"{JOB_PREFIX}{job_id}")
            if job_json:
                jobs.append(json.loads(job_json))
        
        return jobs
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return []
    finally:
        await redis.close()

# Results storage methods
async def store_result(job_id: str, result_data: Dict[str, Any]) -> bool:
    """Store job result in Redis"""
    redis = await get_async_redis_connection()
    try:
        # Convert data to JSON
        result_json = json.dumps(result_data)
        
        # Store in Redis
        await redis.set(f"{RESULT_PREFIX}{job_id}", result_json)
        
        return True
    except Exception as e:
        logger.error(f"Error storing result for job {job_id}: {e}")
        return False
    finally:
        await redis.close()

async def get_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job result from Redis"""
    redis = await get_async_redis_connection()
    try:
        result_json = await redis.get(f"{RESULT_PREFIX}{job_id}")
        if not result_json:
            return None
        
        return json.loads(result_json)
    except Exception as e:
        logger.error(f"Error retrieving result for job {job_id}: {e}")
        return None
    finally:
        await redis.close()

# Cache methods
async def set_cache(key: str, value: Any, expiration: int = 3600) -> bool:
    """Store value in Redis cache with expiration in seconds"""
    redis = await get_async_redis_connection()
    try:
        # Convert data to JSON
        value_json = json.dumps(value)
        
        # Store in Redis with expiration
        await redis.set(f"{CACHE_PREFIX}{key}", value_json, ex=expiration)
        
        return True
    except Exception as e:
        logger.error(f"Error storing cache for {key}: {e}")
        return False
    finally:
        await redis.close()

async def get_cache(key: str) -> Optional[Any]:
    """Get value from Redis cache"""
    redis = await get_async_redis_connection()
    try:
        value_json = await redis.get(f"{CACHE_PREFIX}{key}")
        if not value_json:
            return None
        
        return json.loads(value_json)
    except Exception as e:
        logger.error(f"Error retrieving cache for {key}: {e}")
        return None
    finally:
        await redis.close()

# Statistics methods
async def update_stats(stat_key: str, value: Union[int, float, str]) -> bool:
    """Update system statistics"""
    redis = await get_async_redis_connection()
    try:
        if isinstance(value, (int, float)):
            await redis.set(f"{STATS_PREFIX}{stat_key}", str(value))
        else:
            await redis.set(f"{STATS_PREFIX}{stat_key}", value)
            
        return True
    except Exception as e:
        logger.error(f"Error updating stat {stat_key}: {e}")
        return False
    finally:
        await redis.close()

async def increment_stat(stat_key: str, increment: int = 1) -> int:
    """Increment a statistic counter"""
    redis = await get_async_redis_connection()
    try:
        return await redis.incrby(f"{STATS_PREFIX}{stat_key}", increment)
    except Exception as e:
        logger.error(f"Error incrementing stat {stat_key}: {e}")
        return 0
    finally:
        await redis.close()

async def get_stats() -> Dict[str, Any]:
    """Get all system statistics"""
    redis = await get_async_redis_connection()
    try:
        # Get all keys with stats prefix
        keys = await redis.keys(f"{STATS_PREFIX}*")
        
        # Build stats dictionary
        stats = {}
        for key in keys:
            # Remove prefix to get actual stat name
            stat_name = key[len(STATS_PREFIX):]
            stats[stat_name] = await redis.get(key)
            
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {}
    finally:
        await redis.close()