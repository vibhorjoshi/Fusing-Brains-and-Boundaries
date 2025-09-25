"""
Database utilities for the building footprint API
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import json
import os

import motor.motor_asyncio
from bson import ObjectId
from pymongo import ReturnDocument
import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB ObjectId"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

class DatabaseManager:
    """Database manager for MongoDB and Redis"""
    
    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.redis_client = None
    
    async def initialize(self):
        """Initialize database connections"""
        # MongoDB connection
        self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000
        )
        
        # Get database
        self.db = self.mongo_client[settings.MONGODB_NAME]
        
        # Redis connection
        self.redis_client = redis.from_url(settings.REDIS_URL)
        
        # Test connections
        try:
            # Test MongoDB
            await self.mongo_client.admin.command('ping')
            logger.info("MongoDB connection established")
            
            # Test Redis
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    async def close(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB connection closed")
    
    async def save_job(self, job_data: Dict[str, Any]) -> str:
        """
        Save job data to MongoDB
        
        Args:
            job_data: Job data to save
            
        Returns:
            Job ID
        """
        job_id = job_data.get('job_id', str(ObjectId()))
        job_data['_id'] = job_id
        
        result = await self.db.jobs.update_one(
            {'_id': job_id},
            {'$set': job_data},
            upsert=True
        )
        
        return job_id
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job data from MongoDB
        
        Args:
            job_id: Job ID
            
        Returns:
            Job data or None if not found
        """
        job = await self.db.jobs.find_one({'_id': job_id})
        
        # Convert ObjectId to string
        if job:
            job['_id'] = str(job['_id'])
            
        return job
    
    async def update_job_status(self, job_id: str, status: str) -> bool:
        """
        Update job status in MongoDB
        
        Args:
            job_id: Job ID
            status: New status
            
        Returns:
            Success flag
        """
        result = await self.db.jobs.update_one(
            {'_id': job_id},
            {'$set': {'status': status}}
        )
        
        return result.modified_count > 0
    
    async def save_result(self, job_id: str, result: Dict[str, Any]) -> bool:
        """
        Save job result to MongoDB
        
        Args:
            job_id: Job ID
            result: Result data
            
        Returns:
            Success flag
        """
        # Store result in MongoDB
        result_doc = {
            'job_id': job_id,
            'result': result,
            'created_at': result.get('processing_time', 0)
        }
        
        result = await self.db.results.update_one(
            {'job_id': job_id},
            {'$set': result_doc},
            upsert=True
        )
        
        return True
    
    async def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job result from MongoDB
        
        Args:
            job_id: Job ID
            
        Returns:
            Result data or None if not found
        """
        result = await self.db.results.find_one({'job_id': job_id})
        
        if result:
            # Convert ObjectId to string
            result['_id'] = str(result['_id'])
            
        return result
    
    async def get_jobs(
        self, 
        status: Optional[str] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get jobs from MongoDB
        
        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
            skip: Number of jobs to skip
            
        Returns:
            List of jobs
        """
        # Build query
        query = {}
        if status:
            query['status'] = status
        
        # Get cursor
        cursor = self.db.jobs.find(query).sort('created_at', -1).skip(skip).limit(limit)
        
        # Get jobs
        jobs = []
        async for job in cursor:
            # Convert ObjectId to string
            job['_id'] = str(job['_id'])
            jobs.append(job)
            
        return jobs
    
    async def save_to_cache(self, key: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """
        Save data to Redis cache
        
        Args:
            key: Cache key
            data: Data to save
            ttl: Time to live in seconds
            
        Returns:
            Success flag
        """
        # Convert data to JSON
        json_data = json.dumps(data, cls=JSONEncoder)
        
        # Save to Redis
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._save_to_cache_sync(key, json_data, ttl)
        )
        
        return True
    
    def _save_to_cache_sync(self, key: str, data: str, ttl: int = None):
        """Synchronous helper for saving to Redis cache"""
        if ttl:
            self.redis_client.setex(key, ttl, data)
        else:
            self.redis_client.set(key, data)
    
    async def get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from Redis cache
        
        Args:
            key: Cache key
            
        Returns:
            Data or None if not found
        """
        # Get from Redis
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            self.redis_client.get,
            key
        )
        
        if not data:
            return None
        
        # Convert from JSON
        try:
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error decoding cache data: {str(e)}")
            return None