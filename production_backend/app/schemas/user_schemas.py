"""
User schemas for API request/response validation
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime
from enum import Enum

from app.models.user import UserRole

class UserBase(BaseModel):
    """Base user model"""
    username: str = Field(..., min_length=3, max_length=50, description="Username (3-50 characters)")
    email: EmailStr = Field(..., description="Valid email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name (optional)")

class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (underscores and hyphens allowed)')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        from app.core.security import validate_password_strength
        
        validation_result = validate_password_strength(v)
        if not validation_result["valid"]:
            raise ValueError(f"Password validation failed: {', '.join(validation_result['messages'])}")
        
        return v

class UserLogin(BaseModel):
    """User login model"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")

class UserResponse(UserBase):
    """User response model"""
    id: int
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    """User update model"""
    full_name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None
    
class PasswordChange(BaseModel):
    """Password change model"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        from app.core.security import validate_password_strength
        
        validation_result = validate_password_strength(v)
        if not validation_result["valid"]:
            raise ValueError(f"Password validation failed: {', '.join(validation_result['messages'])}")
        
        return v

class Token(BaseModel):
    """JWT token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Optional[dict] = None
    authentication_method: Optional[str] = "jwt"

class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None

class APIKeyCreate(BaseModel):
    """API key creation model"""
    description: Optional[str] = Field(None, max_length=200, description="Description for the API key")
    expires_days: Optional[int] = Field(None, gt=0, le=365, description="Expiration in days (max 365)")

class APIKeyResponse(BaseModel):
    """API key response model"""
    api_key: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    description: Optional[str] = None

class UserStats(BaseModel):
    """User statistics model"""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_buildings_extracted: int = 0
    account_created: datetime
    last_activity: Optional[datetime] = None