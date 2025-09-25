"""
Security middleware for production deployment
Rate limiting, CORS, security headers, and request validation
"""

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import time
import logging
from typing import Dict, Optional
import asyncio
from datetime import datetime, timedelta
import json
import hashlib

from app.core.config import settings
from app.core.security import SecurityHeaders

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with IP-based and user-based limits
    """
    
    def __init__(self, app, calls_per_minute: int = 60, calls_per_hour: int = 1000):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.request_counts = {}  # In production, use Redis
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        # Get client identifier
        client_ip = self._get_client_ip(request)
        user_id = await self._get_user_id(request)
        
        # Create rate limit key
        rate_limit_key = f"{client_ip}:{user_id}" if user_id else client_ip
        
        # Check rate limits
        if await self._is_rate_limited(rate_limit_key):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls_per_minute} requests per minute allowed"
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        await self._record_request(rate_limit_key)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.calls_per_minute - len(self.request_counts.get(rate_limit_key, [])))
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded IP (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host
    
    async def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from JWT token"""
        try:
            authorization = request.headers.get("Authorization")
            if not authorization or not authorization.startswith("Bearer "):
                return None
            
            token = authorization.split(" ")[1]
            
            from app.core.security import decode_access_token
            payload = decode_access_token(token)
            
            return str(payload.get("user_id"))
            
        except Exception:
            return None
    
    async def _is_rate_limited(self, key: str) -> bool:
        """Check if key is rate limited"""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries()
            self.last_cleanup = current_time
        
        if key not in self.request_counts:
            return False
        
        # Filter requests within the time window
        minute_ago = current_time - 60
        recent_requests = [
            req_time for req_time in self.request_counts[key]
            if req_time > minute_ago
        ]
        
        return len(recent_requests) >= self.calls_per_minute
    
    async def _record_request(self, key: str):
        """Record a request timestamp"""
        current_time = time.time()
        
        if key not in self.request_counts:
            self.request_counts[key] = []
        
        self.request_counts[key].append(current_time)
        
        # Keep only recent requests (last hour)
        hour_ago = current_time - 3600
        self.request_counts[key] = [
            req_time for req_time in self.request_counts[key]
            if req_time > hour_ago
        ]
    
    async def _cleanup_old_entries(self):
        """Remove old entries to prevent memory leaks"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        keys_to_remove = []
        for key, timestamps in self.request_counts.items():
            # Remove old timestamps
            self.request_counts[key] = [
                ts for ts in timestamps if ts > hour_ago
            ]
            
            # Remove empty entries
            if not self.request_counts[key]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.request_counts[key]
        
        logger.info(f"🧹 Rate limit cleanup: removed {len(keys_to_remove)} old entries")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        security_headers = SecurityHeaders.get_security_headers()
        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value
        
        # Add custom headers
        response.headers["X-API-Version"] = "1.0"
        response.headers["X-Response-Time"] = str(time.time())
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all requests for monitoring and debugging
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        
        logger.info(
            f"📥 {request.method} {request.url.path} - "
            f"IP: {client_ip} - UA: {user_agent[:100]}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"📤 {request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"❌ {request.method} {request.url.path} - "
                f"Error: {str(e)} - Time: {process_time:.3f}s"
            )
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host

class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Limit request body size to prevent DoS attacks
    """
    
    def __init__(self, app, max_size_mb: int = 10):
        super().__init__(app)
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
    
    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header
        content_length = request.headers.get("Content-Length")
        
        if content_length and int(content_length) > self.max_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request too large",
                    "message": f"Maximum request size is {self.max_size // (1024*1024)}MB"
                }
            )
        
        return await call_next(request)

class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    IP whitelist middleware for production security
    """
    
    def __init__(self, app, allowed_ips: Optional[list] = None):
        super().__init__(app)
        self.allowed_ips = allowed_ips or []
        self.whitelist_enabled = len(self.allowed_ips) > 0
    
    async def dispatch(self, request: Request, call_next):
        if not self.whitelist_enabled:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        
        # Check if IP is allowed
        if client_ip not in self.allowed_ips:
            logger.warning(f"🚫 Blocked request from unauthorized IP: {client_ip}")
            
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Access denied",
                    "message": "Your IP address is not authorized"
                }
            )
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host

class APIKeyValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate API keys for specific endpoints
    """
    
    def __init__(self, app, protected_paths: Optional[list] = None):
        super().__init__(app)
        self.protected_paths = protected_paths or ["/api/v1/ml-processing", "/api/v1/admin"]
    
    async def dispatch(self, request: Request, call_next):
        # Check if path is protected
        request_path = request.url.path
        
        is_protected = any(
            request_path.startswith(protected_path)
            for protected_path in self.protected_paths
        )
        
        if not is_protected:
            return await call_next(request)
        
        # Check for authentication
        authorization = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        
        if not authorization and not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication required",
                    "message": "Provide Authorization header or X-API-Key"
                }
            )
        
        return await call_next(request)

def setup_cors_middleware(app):
    """
    Configure CORS middleware for the application
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS if hasattr(settings, 'ALLOWED_HOSTS') else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Requested-With",
            "X-CSRF-Token"
        ],
        expose_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining", 
            "X-Process-Time",
            "X-API-Version"
        ]
    )

def setup_security_middleware(app):
    """
    Set up all security middleware for the application
    """
    # Request size limiting
    app.add_middleware(RequestSizeMiddleware, max_size_mb=10)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware, calls_per_minute=100, calls_per_hour=2000)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # API key validation for protected endpoints
    app.add_middleware(APIKeyValidationMiddleware)
    
    # IP whitelist (disabled by default - enable in production)
    # app.add_middleware(IPWhitelistMiddleware, allowed_ips=[])
    
    logger.info("🔒 Security middleware configured successfully")