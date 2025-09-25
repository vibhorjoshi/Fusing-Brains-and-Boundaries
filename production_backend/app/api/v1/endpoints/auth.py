"""
Authentication API endpoints
Production authentication system with JWT and API keys
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any

from app.core.database import get_db
from app.models.user import User, UserRole
from app.core.security import (
    verify_password, get_password_hash, create_access_token,
    decode_access_token, create_api_key, verify_api_key
)
from app.schemas.user_schemas import (
    UserLogin, UserCreate, UserResponse, Token,
    PasswordChange, APIKeyCreate, APIKeyResponse
)
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)
security = HTTPBearer()

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account
    
    Creates a new user with hashed password and API key generation.
    Default role is USER, admin registration requires separate endpoint.
    """
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.email == user_data.email) | (User.username == user_data.username)
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User with this email or username already exists"
            )
        
        # Create new user
        user = User(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            password_hash=get_password_hash(user_data.password),
            role=UserRole.USER,
            is_active=True
        )
        
        # Generate API key for new user
        api_key = create_api_key(user.username)
        user.api_key = api_key
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        logger.info(f"✅ New user registered: {user.username}")
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at
        )
        
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Registration failed")

@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT token
    
    Validates credentials and returns access token for API authentication.
    """
    try:
        # Find user by email or username
        user = db.query(User).filter(
            (User.email == user_credentials.username) | 
            (User.username == user_credentials.username)
        ).first()
        
        if not user or not verify_password(user_credentials.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is inactive"
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role.value},
            expires_delta=access_token_expires
        )
        
        logger.info(f"✅ User login successful: {user.username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user_info": {
                "username": user.username,
                "role": user.role.value,
                "full_name": user.full_name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Refresh JWT access token
    
    Validates current token and issues new token with extended expiration.
    """
    try:
        # Decode current token
        payload = decode_access_token(credentials.credentials)
        username = payload.get("sub")
        
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Verify user still exists and is active
        user = db.query(User).filter(User.username == username).first()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role.value},
            expires_delta=access_token_expires
        )
        
        logger.info(f"✅ Token refreshed for user: {username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Change user password
    
    Validates current password and updates to new password with proper hashing.
    """
    try:
        # Get current user from token
        payload = decode_access_token(credentials.credentials)
        username = payload.get("sub")
        
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not verify_password(password_data.current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        user.password_hash = get_password_hash(password_data.new_password)
        user.password_changed_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"✅ Password changed for user: {username}")
        
        return {"status": "success", "message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Password change failed: {e}")
        raise HTTPException(status_code=500, detail="Password change failed")

@router.post("/generate-api-key", response_model=APIKeyResponse)
async def generate_api_key(
    api_key_data: APIKeyCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Generate new API key for user
    
    Creates a new API key for programmatic access with optional expiration.
    """
    try:
        # Get current user
        payload = decode_access_token(credentials.credentials)
        username = payload.get("sub")
        
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate new API key
        api_key = create_api_key(username, api_key_data.description)
        
        # Update user record
        user.api_key = api_key
        user.api_key_created_at = datetime.utcnow()
        
        if api_key_data.expires_days:
            user.api_key_expires_at = datetime.utcnow() + timedelta(days=api_key_data.expires_days)
        
        db.commit()
        
        logger.info(f"✅ New API key generated for user: {username}")
        
        return APIKeyResponse(
            api_key=api_key,
            created_at=user.api_key_created_at,
            expires_at=user.api_key_expires_at,
            description=api_key_data.description
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API key generation failed: {e}")
        raise HTTPException(status_code=500, detail="API key generation failed")

@router.delete("/revoke-api-key")
async def revoke_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Revoke user's API key
    
    Invalidates the current API key for security purposes.
    """
    try:
        # Get current user
        payload = decode_access_token(credentials.credentials)
        username = payload.get("sub")
        
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Revoke API key
        user.api_key = None
        user.api_key_created_at = None
        user.api_key_expires_at = None
        
        db.commit()
        
        logger.info(f"✅ API key revoked for user: {username}")
        
        return {"status": "success", "message": "API key revoked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API key revocation failed: {e}")
        raise HTTPException(status_code=500, detail="API key revocation failed")

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Get current user information
    
    Returns detailed information about the authenticated user.
    """
    try:
        # Get current user
        payload = decode_access_token(credentials.credentials)
        username = payload.get("sub")
        
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get user info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user information")

@router.post("/validate-token")
async def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Validate JWT token
    
    Verifies token validity and returns token information.
    """
    try:
        payload = decode_access_token(credentials.credentials)
        
        return {
            "valid": True,
            "username": payload.get("sub"),
            "user_id": payload.get("user_id"),
            "role": payload.get("role"),
            "expires": payload.get("exp")
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

@router.post("/api-key-login", response_model=Token)
async def api_key_login(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Authenticate using API key
    
    Alternative authentication method using API key instead of username/password.
    """
    try:
        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        # Verify API key
        username = verify_api_key(api_key, db)
        
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Get user
        user = db.query(User).filter(User.username == username).first()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role.value},
            expires_delta=access_token_expires
        )
        
        logger.info(f"✅ API key authentication successful: {username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "authentication_method": "api_key"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API key authentication failed: {e}")
        raise HTTPException(status_code=500, detail="API key authentication failed")