"""
Authentication Models for GeoAI Research Backend
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher" 
    VIEWER = "viewer"


class ComponentType(str, Enum):
    MAP_PROCESSING = "MapProcessing"
    ADAPTIVE_FUSION = "AdaptiveFusion"
    VECTOR_CONVERSION = "VectorConversion"
    GRAPH_VISUALIZATION = "GraphVisualization"
    ML_MODEL_ACCESS = "MLModelAccess"
    SYSTEM_ADMIN = "SystemAdmin"


class User(BaseModel):
    """User data model"""
    id: Optional[int] = None
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKey(BaseModel):
    """API Key data model"""
    id: Optional[int] = None
    key: str
    component_type: ComponentType
    user_id: int
    is_active: bool = True
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Session(BaseModel):
    """User session data model"""
    id: Optional[int] = None
    user_id: int
    session_token: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """Registration request model"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class APIKeyRequest(BaseModel):
    """API Key validation request"""
    api_key: str
    component_name: Optional[str] = None


class AuthResponse(BaseModel):
    """Authentication response model"""
    success: bool
    message: str
    user: Optional[User] = None
    token: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }