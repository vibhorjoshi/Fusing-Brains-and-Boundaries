"""
Configuration for Unified Backend
Handles both development and production settings
"""

import os
from pathlib import Path
from typing import List, Optional

class UnifiedConfig:
    """Configuration class for the unified backend"""
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # API Settings
    API_TITLE: str = "Building Footprint AI - Unified Backend"
    API_DESCRIPTION: str = "Production-ready backend for building footprint extraction"
    API_VERSION: str = "1.0.0"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # File Processing
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 104857600))  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
    TEMP_UPLOAD_DIR: str = os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")
    
    # ML Pipeline Settings
    ML_MODEL_PATH: Optional[str] = os.getenv("ML_MODEL_PATH")
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", 5))
    JOB_TIMEOUT_MINUTES: int = int(os.getenv("JOB_TIMEOUT_MINUTES", 30))
    
    # Database Settings (Production)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # Redis Settings (Production)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    
    # AWS Settings (Production)
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ]
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory for file uploads"""
        temp_dir = Path(self.TEMP_UPLOAD_DIR)
        temp_dir.mkdir(exist_ok=True)
        return temp_dir

# Global config instance
config = UnifiedConfig()