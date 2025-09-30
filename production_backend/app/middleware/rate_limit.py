"""
Rate Limiting Middleware  
Implements rate limiting using Redis for distributed applications
"""

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.redis_manager import get_redis
import logging

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis"""
    
    def __init__(self, app, requests_per_minute: int = 100, requests_per_hour: int = 1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
    
    async def dispatch(self, request: Request, call_next):
        """Check rate limits before processing request"""
        
        # Skip rate limiting for certain paths
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"
        
        try:
            redis_manager = await get_redis()
            
            # Check minute-based rate limit
            minute_key = f"{client_ip}:minute"
            minute_result = await redis_manager.check_rate_limit(
                minute_key, self.requests_per_minute, 60
            )
            
            if not minute_result["allowed"]:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Too many requests per minute.",
                    headers={
                        "X-RateLimit-Remaining": str(minute_result["remaining"]),
                        "X-RateLimit-Reset": str(minute_result["reset_time"])
                    }
                )
            
            # Check hour-based rate limit
            hour_key = f"{client_ip}:hour"
            hour_result = await redis_manager.check_rate_limit(
                hour_key, self.requests_per_hour, 3600
            )
            
            if not hour_result["allowed"]:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Too many requests per hour.",
                    headers={
                        "X-RateLimit-Remaining": str(hour_result["remaining"]),
                        "X-RateLimit-Reset": str(hour_result["reset_time"])
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining-Minute"] = str(minute_result["remaining"])
            response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
            response.headers["X-RateLimit-Remaining-Hour"] = str(hour_result["remaining"])
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Rate limiting failed for {client_ip}: {e}")
            # If Redis is down, allow request to proceed
            return await call_next(request)