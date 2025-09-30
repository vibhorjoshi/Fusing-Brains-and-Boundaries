"""
Authentication Controller for GeoAI Research Backend
Handles all authentication-related business logic
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, status

from ..models.auth_models import (
    User, APIKey, Session, LoginRequest, RegisterRequest,
    APIKeyRequest, AuthResponse, ComponentType
)
from ..services.auth_service import AuthService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AuthController:
    """Authentication controller handling auth business logic"""
    
    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service
        
        # Predefined API keys for components (in production, store in database)
        self.valid_api_keys = {
            'GEO_SAT_PROC_2024_001': ComponentType.MAP_PROCESSING,
            'ADAPT_FUSION_AI_2024_002': ComponentType.ADAPTIVE_FUSION,
            'VECTOR_CONV_SYS_2024_003': ComponentType.VECTOR_CONVERSION,
            'GRAPH_VIZ_ENGINE_2024_004': ComponentType.GRAPH_VISUALIZATION,
            'ML_MODEL_ACCESS_2024_005': ComponentType.ML_MODEL_ACCESS,
            'ADMIN_CONTROL_2024_006': ComponentType.SYSTEM_ADMIN
        }
    
    async def register_user(self, request: RegisterRequest) -> AuthResponse:
        """Register a new user"""
        try:
            logger.info(f"Attempting to register user: {request.username}")
            
            # Check if user already exists
            existing_user = await self.auth_service.get_user_by_username(request.username)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists"
                )
            
            existing_email = await self.auth_service.get_user_by_email(request.email)
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Hash password
            password_hash = self._hash_password(request.password)
            
            # Create user
            user = User(
                username=request.username,
                email=request.email,
                full_name=request.full_name,
                created_at=datetime.now()
            )
            
            created_user = await self.auth_service.create_user(user, password_hash)
            
            # Generate session token
            session_token = self._generate_session_token()
            session = Session(
                user_id=created_user.id,
                session_token=session_token,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=24)
            )
            
            await self.auth_service.create_session(session)
            
            logger.info(f"User registered successfully: {request.username}")
            
            return AuthResponse(
                success=True,
                message="User registered successfully",
                user=created_user,
                token=session_token,
                expires_at=session.expires_at
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Registration failed for {request.username}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    async def login_user(self, request: LoginRequest) -> AuthResponse:
        """Authenticate user login"""
        try:
            logger.info(f"Login attempt for user: {request.username}")
            
            # Get user and password hash
            user, password_hash = await self.auth_service.get_user_with_password(request.username)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password"
                )
            
            # Verify password
            if not self._verify_password(request.password, password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password"
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is deactivated"
                )
            
            # Generate session token
            session_token = self._generate_session_token()
            session = Session(
                user_id=user.id,
                session_token=session_token,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=24)
            )
            
            await self.auth_service.create_session(session)
            await self.auth_service.update_last_login(user.id)
            
            logger.info(f"User logged in successfully: {request.username}")
            
            return AuthResponse(
                success=True,
                message="Login successful",
                user=user,
                token=session_token,
                expires_at=session.expires_at
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Login failed for {request.username}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Login failed"
            )
    
    async def validate_api_key(self, request: APIKeyRequest) -> Dict[str, any]:
        """Validate API key for component access"""
        try:
            api_key = request.api_key
            
            if api_key in self.valid_api_keys:
                component_type = self.valid_api_keys[api_key]
                
                # Log API key usage
                await self.auth_service.log_api_key_usage(api_key)
                
                logger.info(f"API key validated for component: {component_type.value}")
                
                return {
                    "status": "valid",
                    "component": component_type.value,
                    "access_level": "authenticated",
                    "timestamp": time.time()
                }
            else:
                logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
                return {
                    "status": "invalid",
                    "error": "Invalid API key",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"API key validation error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def validate_session(self, session_token: str) -> Optional[User]:
        """Validate session token"""
        try:
            session = await self.auth_service.get_session(session_token)
            if not session or not session.is_active:
                return None
            
            if session.expires_at < datetime.now():
                await self.auth_service.deactivate_session(session_token)
                return None
            
            user = await self.auth_service.get_user_by_id(session.user_id)
            return user if user and user.is_active else None
            
        except Exception as e:
            logger.error(f"Session validation error: {str(e)}")
            return None
    
    async def logout_user(self, session_token: str) -> bool:
        """Logout user by deactivating session"""
        try:
            await self.auth_service.deactivate_session(session_token)
            logger.info("User logged out successfully")
            return True
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password using PBKDF2"""
        salt = secrets.token_hex(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return salt + pwdhash.hex()
    
    def _verify_password(self, password: str, stored_password: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt = stored_password[:64]
            stored_hash = stored_password[64:]
            pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            return hmac.compare_digest(pwdhash.hex(), stored_hash)
        except Exception:
            return False
    
    def _generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    async def get_user_permissions(self, user_id: int) -> Dict[str, any]:
        """Get user permissions and access levels"""
        try:
            user = await self.auth_service.get_user_by_id(user_id)
            if not user:
                return {"error": "User not found"}
            
            # Define role-based permissions
            permissions = {
                "admin": ["read", "write", "delete", "manage_users", "system_admin"],
                "researcher": ["read", "write", "process_data", "train_models"],
                "viewer": ["read", "view_results"]
            }
            
            user_permissions = permissions.get(user.role.value, ["read"])
            
            return {
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
                "permissions": user_permissions,
                "is_active": user.is_active
            }
            
        except Exception as e:
            logger.error(f"Error getting user permissions: {str(e)}")
            return {"error": str(e)}