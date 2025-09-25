"""
Configuration management for Building Footprint API
"""

from typing import Optional, Dict, Any, List
import os
from functools import lru_cache

class Settings:
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    API_TITLE: str = "Building Footprint Extraction API"
    API_DESCRIPTION: str = "API for extracting building footprints from satellite imagery"
    API_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    PROJECT_NAME: str = "Building Footprint API"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Security
    SECRET_KEY: str = "dev-secret-key-for-testing-only"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    REDIS_URL: str = "redis://localhost:6379/0"
    MONGODB_URL: str = "mongodb://localhost:27017/building_footprint_db"
    MONGODB_NAME: str = "building_footprint_db"
    RESULT_TTL: int = 86400  # 24 hours in seconds
    DATABASE_URL: Optional[str] = None
    
    # Google Maps API
    MAPS_API_KEY: str = "AIzaSyDevelopmentKeyPlaceholder"
    GOOGLE_MAPS_API_KEY: str = "AIzaSyDevelopmentKeyPlaceholder"
    
    # Model Settings
    MODEL_PATH: str = "./models"
    MAX_IMAGE_SIZE: int = 2048
    
    # Processing Settings
    MAX_CONCURRENT_JOBS: int = 4
    JOB_TIMEOUT: int = 600  # 10 minutes
    
    # File Paths
    LOG_DIR: str = "./logs"
    TEMP_DIR: str = "./temp"
    DATA_DIR: str = "./data"
    
    # Storage
    STORAGE_TYPE: str = "local"  # local, s3, gcs
    STORAGE_BUCKET: Optional[str] = None
    
    def __init__(self):
        """Initialize settings with environment variable overrides"""
        # Override with environment variables if they exist
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                env_value = os.getenv(attr_name)
                if env_value is not None:
                    # Convert string env vars to appropriate types
                    current_value = getattr(self, attr_name)
                    if isinstance(current_value, bool):
                        setattr(self, attr_name, env_value.lower() in ('true', '1', 'yes'))
                    elif isinstance(current_value, int):
                        try:
                            setattr(self, attr_name, int(env_value))
                        except ValueError:
                            pass  # Keep default value if conversion fails
                    elif isinstance(current_value, list):
                        # Handle comma-separated list values
                        setattr(self, attr_name, env_value.split(','))
                    else:
                        setattr(self, attr_name, env_value)

@lru_cache()
def get_settings():
    """Get settings with caching for performance"""
    return Settings()