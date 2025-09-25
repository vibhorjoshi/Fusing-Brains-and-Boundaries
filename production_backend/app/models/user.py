"""
User management models
Authentication and authorization for the production system
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash
import enum
from datetime import datetime, timedelta
import jwt
from typing import Optional, Dict, Any

from app.core.database import Base
from app.core.config import settings

class UserRole(str, enum.Enum):
    """User roles for role-based access control"""
    ADMIN = "admin"
    RESEARCHER = "researcher" 
    API_USER = "api_user"
    VIEWER = "viewer"

class UserStatus(str, enum.Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Role and permissions
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    status = Column(Enum(UserStatus), default=UserStatus.PENDING, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # API access
    api_key = Column(String(255), unique=True, index=True, nullable=True)
    api_key_expires_at = Column(DateTime, nullable=True)
    rate_limit_per_minute = Column(Integer, default=100)
    rate_limit_per_hour = Column(Integer, default=1000)
    
    # Organization info
    organization = Column(String(200), nullable=True)
    department = Column(String(100), nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)  # Store additional user data
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    processing_jobs = relationship("ProcessingJob", back_populates="user")
    file_uploads = relationship("FileStorage", back_populates="user")
    api_usage = relationship("APIUsage", back_populates="user")
    
    def set_password(self, password: str):
        """Hash and set user password"""
        self.hashed_password = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Check if provided password matches hash"""
        return check_password_hash(self.hashed_password, password)
    
    def generate_api_key(self) -> str:
        """Generate a new API key for the user"""
        import secrets
        api_key = f"bfai_{secrets.token_urlsafe(32)}"
        self.api_key = api_key
        # API key expires in 1 year
        self.api_key_expires_at = datetime.utcnow() + timedelta(days=365)
        return api_key
    
    def is_api_key_valid(self) -> bool:
        """Check if user's API key is still valid"""
        if not self.api_key or not self.api_key_expires_at:
            return False
        return datetime.utcnow() < self.api_key_expires_at
    
    def generate_access_token(self) -> str:
        """Generate JWT access token"""
        payload = {
            "user_id": self.id,
            "username": self.username,
            "role": self.role.value,
            "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    
    def generate_refresh_token(self) -> str:
        """Generate JWT refresh token"""
        payload = {
            "user_id": self.id,
            "username": self.username,
            "exp": datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        role_permissions = {
            UserRole.ADMIN: ["*"],  # Admin has all permissions
            UserRole.RESEARCHER: [
                "building_footprint.read",
                "building_footprint.create", 
                "building_footprint.process",
                "files.upload",
                "files.read",
                "jobs.create",
                "jobs.read"
            ],
            UserRole.API_USER: [
                "building_footprint.read",
                "building_footprint.create",
                "files.upload",
                "files.read"
            ],
            UserRole.VIEWER: [
                "building_footprint.read",
                "files.read"
            ]
        }
        
        permissions = role_permissions.get(self.role, [])
        return "*" in permissions or permission in permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (for API responses)"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value,
            "status": self.status.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "organization": self.organization,
            "department": self.department,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "api_key_expires_at": self.api_key_expires_at.isoformat() if self.api_key_expires_at else None
        }

class APIUsage(Base):
    """Track API usage for rate limiting and analytics"""
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Request details
    endpoint = Column(String(200), nullable=False)
    method = Column(String(10), nullable=False)
    ip_address = Column(String(45), nullable=True)  # Support IPv6
    user_agent = Column(Text, nullable=True)
    
    # Response details
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer, nullable=True)
    
    # Metadata
    request_size_bytes = Column(Integer, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationship
    user = relationship("User", back_populates="api_usage")

class UserSession(Base):
    """Track user sessions for security and analytics"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Session details
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Session lifecycle
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    last_activity = Column(DateTime, default=func.now())
    
    @property
    def is_valid(self) -> bool:
        """Check if session is still valid"""
        return (
            self.revoked_at is None and
            datetime.utcnow() < self.expires_at
        )
    
    def revoke(self):
        """Revoke the session"""
        self.revoked_at = datetime.utcnow()