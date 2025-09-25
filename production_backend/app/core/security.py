"""
Security utilities
JWT tokens, password hashing, and API key management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets
import hashlib
import logging
from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def get_password_hash(password: str) -> str:
    """
    Generate password hash
    """
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Token payload data
        expires_delta: Token expiration time
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    try:
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.SECRET_KEY, 
            algorithm=settings.ALGORITHM
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise

def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT access token
    
    Args:
        token: JWT token string
    
    Returns:
        Token payload data
    
    Raises:
        JWTError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        # Validate token type
        if payload.get("type") != "access":
            raise JWTError("Invalid token type")
        
        return payload
        
    except JWTError as e:
        logger.warning(f"Token decode failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {e}")
        raise JWTError("Token decode error")

def create_api_key(username: str, description: Optional[str] = None) -> str:
    """
    Create API key for user
    
    Args:
        username: User's username
        description: Optional description for the API key
    
    Returns:
        Generated API key string
    """
    try:
        # Generate random part
        random_part = secrets.token_urlsafe(32)
        
        # Create key with username prefix
        timestamp = str(int(datetime.utcnow().timestamp()))
        key_data = f"{username}:{timestamp}:{random_part}"
        
        # Hash the key data
        api_key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Format as API key (prefix + hash)
        api_key = f"gai_{api_key_hash[:48]}"
        
        logger.info(f"✅ API key created for user: {username}")
        return api_key
        
    except Exception as e:
        logger.error(f"API key creation failed: {e}")
        raise

def verify_api_key(api_key: str, db: Session) -> Optional[str]:
    """
    Verify API key and return username
    
    Args:
        api_key: API key string to verify
        db: Database session
    
    Returns:
        Username if key is valid, None otherwise
    """
    try:
        # Check key format
        if not api_key.startswith("gai_") or len(api_key) != 52:
            return None
        
        from app.models.user import User
        from datetime import datetime
        
        # Find user with this API key
        user = db.query(User).filter(User.api_key == api_key).first()
        
        if not user:
            return None
        
        # Check if user is active
        if not user.is_active:
            return None
        
        # Check if API key is expired
        if user.api_key_expires_at and user.api_key_expires_at < datetime.utcnow():
            return None
        
        return user.username
        
    except Exception as e:
        logger.error(f"API key verification failed: {e}")
        return None

def generate_reset_token(user_id: int) -> str:
    """
    Generate password reset token
    
    Args:
        user_id: User's database ID
    
    Returns:
        Reset token string
    """
    data = {
        "user_id": user_id,
        "type": "reset",
        "exp": datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
    }
    
    try:
        token = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return token
    except Exception as e:
        logger.error(f"Reset token generation failed: {e}")
        raise

def verify_reset_token(token: str) -> Optional[int]:
    """
    Verify password reset token
    
    Args:
        token: Reset token string
    
    Returns:
        User ID if token is valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        if payload.get("type") != "reset":
            return None
        
        return payload.get("user_id")
        
    except JWTError:
        return None
    except Exception as e:
        logger.error(f"Reset token verification failed: {e}")
        return None

class SecurityHeaders:
    """
    Security headers for HTTP responses
    """
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """
        Get standard security headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Validate password strength
    
    Args:
        password: Password to validate
    
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "score": 0,
        "messages": []
    }
    
    # Length check
    if len(password) < 8:
        result["valid"] = False
        result["messages"].append("Password must be at least 8 characters long")
    else:
        result["score"] += 1
    
    # Complexity checks
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    if has_upper:
        result["score"] += 1
    else:
        result["messages"].append("Password should contain uppercase letters")
    
    if has_lower:
        result["score"] += 1
    else:
        result["messages"].append("Password should contain lowercase letters")
    
    if has_digit:
        result["score"] += 1
    else:
        result["messages"].append("Password should contain numbers")
    
    if has_special:
        result["score"] += 1
    else:
        result["messages"].append("Password should contain special characters")
    
    # Common password check (basic)
    common_passwords = [
        "password", "123456", "password123", "admin", "qwerty",
        "letmein", "welcome", "monkey", "dragon", "master"
    ]
    
    if password.lower() in common_passwords:
        result["valid"] = False
        result["messages"].append("Password is too common")
    
    # Final validation
    if result["score"] < 3:
        result["valid"] = False
    
    return result

def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """
    Mask sensitive data for logging
    
    Args:
        data: Sensitive data to mask
        mask_char: Character to use for masking
        visible_chars: Number of characters to keep visible
    
    Returns:
        Masked string
    """
    if not data or len(data) <= visible_chars:
        return mask_char * len(data) if data else ""
    
    return data[:visible_chars] + mask_char * (len(data) - visible_chars)