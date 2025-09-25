"""
Authentication dependencies
Dependency injection for authentication and authorization
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.core.database import get_db
from app.models.user import User, UserRole
from app.core.security import decode_access_token, verify_api_key

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None,
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user
    
    Supports both JWT token and API key authentication methods.
    """
    user = None
    
    # Try JWT token authentication first
    if credentials:
        try:
            payload = decode_access_token(credentials.credentials)
            username = payload.get("sub")
            
            if username:
                user = db.query(User).filter(User.username == username).first()
                
        except Exception as e:
            logger.warning(f"JWT authentication failed: {e}")
    
    # Fallback to API key authentication
    if not user and request:
        api_key = request.headers.get("X-API-Key")
        
        if api_key:
            try:
                username = verify_api_key(api_key, db)
                
                if username:
                    user = db.query(User).filter(User.username == username).first()
                    
            except Exception as e:
                logger.warning(f"API key authentication failed: {e}")
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive"
        )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user (alias for compatibility)
    """
    return current_user

async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user with admin role requirement
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user

async def get_premium_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user with premium or admin role requirement
    """
    if current_user.role not in [UserRole.PREMIUM, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium or Admin access required"
        )
    
    return current_user

def require_role(required_role: UserRole):
    """
    Create a dependency that requires a specific user role
    
    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(user: User = Depends(require_role(UserRole.ADMIN))):
            ...
    """
    async def role_dependency(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {required_role.value} required"
            )
        return current_user
    
    return role_dependency

def require_roles(*required_roles: UserRole):
    """
    Create a dependency that requires one of several user roles
    
    Usage:
        @router.get("/premium-or-admin")
        async def premium_endpoint(user: User = Depends(require_roles(UserRole.PREMIUM, UserRole.ADMIN))):
            ...
    """
    async def roles_dependency(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if current_user.role not in required_roles:
            roles_str = ", ".join([role.value for role in required_roles])
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these roles required: {roles_str}"
            )
        return current_user
    
    return roles_dependency

class RateLimitDependency:
    """
    Rate limiting dependency for API endpoints
    """
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self._cache = {}
    
    async def __call__(
        self,
        request: Request,
        current_user: User = Depends(get_current_user)
    ):
        """
        Check rate limits for the current user
        
        Args:
            calls: Number of calls allowed
            period: Time period in seconds
        """
        from time import time
        
        user_id = current_user.id
        current_time = time()
        
        # Clean old entries
        if user_id in self._cache:
            self._cache[user_id] = [
                timestamp for timestamp in self._cache[user_id]
                if current_time - timestamp < self.period
            ]
        else:
            self._cache[user_id] = []
        
        # Check if limit exceeded
        if len(self._cache[user_id]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.calls} calls per {self.period} seconds"
            )
        
        # Add current call
        self._cache[user_id].append(current_time)

# Pre-configured rate limit dependencies
rate_limit_default = RateLimitDependency(calls=100, period=3600)  # 100 calls per hour
rate_limit_strict = RateLimitDependency(calls=10, period=60)      # 10 calls per minute
rate_limit_processing = RateLimitDependency(calls=5, period=300)  # 5 calls per 5 minutes

async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Optional authentication dependency
    
    Returns user if authenticated, None if not authenticated.
    Does not raise exceptions for missing authentication.
    """
    try:
        return await get_current_user(credentials, request, db)
    except HTTPException:
        return None