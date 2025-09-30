"""
Authentication Service for GeoAI Research Backend
Handles database operations for authentication
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from ..models.auth_models import User, APIKey, Session
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AuthService:
    """Authentication service for database operations"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # File-based storage for demo (in production, use proper database)
        self.users_file = os.path.join(data_dir, "users.json")
        self.sessions_file = os.path.join(data_dir, "sessions.json")
        self.api_keys_file = os.path.join(data_dir, "api_keys.json")
        
        self._init_storage_files()
    
    def _init_storage_files(self):
        """Initialize storage files if they don't exist"""
        for file_path in [self.users_file, self.sessions_file, self.api_keys_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump({}, f)
    
    async def create_user(self, user: User, password_hash: str) -> User:
        """Create a new user"""
        try:
            users_data = self._load_json(self.users_file)
            
            # Generate user ID
            user_id = max([int(k) for k in users_data.keys()] + [0]) + 1
            user.id = user_id
            
            # Store user data
            users_data[str(user_id)] = {
                **user.dict(),
                "password_hash": password_hash,
                "created_at": user.created_at.isoformat() if user.created_at else None
            }
            
            self._save_json(self.users_file, users_data)
            logger.info(f"User created with ID: {user_id}")
            
            return user
            
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            users_data = self._load_json(self.users_file)
            
            for user_id, user_data in users_data.items():
                if user_data.get("username") == username:
                    user_dict = {k: v for k, v in user_data.items() if k != "password_hash"}
                    if user_dict.get("created_at"):
                        user_dict["created_at"] = datetime.fromisoformat(user_dict["created_at"])
                    if user_dict.get("last_login"):
                        user_dict["last_login"] = datetime.fromisoformat(user_dict["last_login"])
                    
                    return User(**user_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            users_data = self._load_json(self.users_file)
            
            for user_id, user_data in users_data.items():
                if user_data.get("email") == email:
                    user_dict = {k: v for k, v in user_data.items() if k != "password_hash"}
                    if user_dict.get("created_at"):
                        user_dict["created_at"] = datetime.fromisoformat(user_dict["created_at"])
                    if user_dict.get("last_login"):
                        user_dict["last_login"] = datetime.fromisoformat(user_dict["last_login"])
                    
                    return User(**user_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            users_data = self._load_json(self.users_file)
            user_data = users_data.get(str(user_id))
            
            if user_data:
                user_dict = {k: v for k, v in user_data.items() if k != "password_hash"}
                if user_dict.get("created_at"):
                    user_dict["created_at"] = datetime.fromisoformat(user_dict["created_at"])
                if user_dict.get("last_login"):
                    user_dict["last_login"] = datetime.fromisoformat(user_dict["last_login"])
                
                return User(**user_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    async def get_user_with_password(self, username: str) -> Tuple[Optional[User], Optional[str]]:
        """Get user with password hash for authentication"""
        try:
            users_data = self._load_json(self.users_file)
            
            for user_id, user_data in users_data.items():
                if user_data.get("username") == username:
                    user_dict = {k: v for k, v in user_data.items() if k != "password_hash"}
                    if user_dict.get("created_at"):
                        user_dict["created_at"] = datetime.fromisoformat(user_dict["created_at"])
                    if user_dict.get("last_login"):
                        user_dict["last_login"] = datetime.fromisoformat(user_dict["last_login"])
                    
                    user = User(**user_dict)
                    password_hash = user_data.get("password_hash")
                    
                    return user, password_hash
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error getting user with password: {str(e)}")
            return None, None
    
    async def create_session(self, session: Session) -> Session:
        """Create a new session"""
        try:
            sessions_data = self._load_json(self.sessions_file)
            
            session_data = session.dict()
            if session_data.get("created_at"):
                session_data["created_at"] = session_data["created_at"].isoformat()
            if session_data.get("expires_at"):
                session_data["expires_at"] = session_data["expires_at"].isoformat()
            
            sessions_data[session.session_token] = session_data
            self._save_json(self.sessions_file, sessions_data)
            
            logger.info(f"Session created for user: {session.user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise
    
    async def get_session(self, session_token: str) -> Optional[Session]:
        """Get session by token"""
        try:
            sessions_data = self._load_json(self.sessions_file)
            session_data = sessions_data.get(session_token)
            
            if session_data:
                if session_data.get("created_at"):
                    session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
                if session_data.get("expires_at"):
                    session_data["expires_at"] = datetime.fromisoformat(session_data["expires_at"])
                
                return Session(**session_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            return None
    
    async def deactivate_session(self, session_token: str) -> bool:
        """Deactivate a session"""
        try:
            sessions_data = self._load_json(self.sessions_file)
            
            if session_token in sessions_data:
                sessions_data[session_token]["is_active"] = False
                self._save_json(self.sessions_file, sessions_data)
                logger.info(f"Session deactivated: {session_token[:10]}...")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deactivating session: {str(e)}")
            return False
    
    async def update_last_login(self, user_id: int) -> bool:
        """Update user's last login time"""
        try:
            users_data = self._load_json(self.users_file)
            
            if str(user_id) in users_data:
                users_data[str(user_id)]["last_login"] = datetime.now().isoformat()
                self._save_json(self.users_file, users_data)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating last login: {str(e)}")
            return False
    
    async def log_api_key_usage(self, api_key: str) -> bool:
        """Log API key usage"""
        try:
            api_keys_data = self._load_json(self.api_keys_file)
            
            if api_key not in api_keys_data:
                api_keys_data[api_key] = {
                    "usage_count": 0,
                    "last_used": None,
                    "created_at": datetime.now().isoformat()
                }
            
            api_keys_data[api_key]["usage_count"] += 1
            api_keys_data[api_key]["last_used"] = datetime.now().isoformat()
            
            self._save_json(self.api_keys_file, api_keys_data)
            return True
            
        except Exception as e:
            logger.error(f"Error logging API key usage: {str(e)}")
            return False
    
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_json(self, file_path: str, data: Dict[str, Any]) -> None:
        """Save JSON data to file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)