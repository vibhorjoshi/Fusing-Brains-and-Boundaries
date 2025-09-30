"""
Configuration settings for GeoAI Research Backend
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    app_name: str = "GeoAI Research Backend"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8002
    reload: bool = True
    
    # Security settings
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # CORS settings
    allowed_origins: List[str] = ["*"]
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]
    
    # Database settings (for future use)
    database_url: Optional[str] = None
    database_pool_size: int = 5
    
    # Redis settings (for caching)
    redis_url: Optional[str] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    # File storage settings
    data_dir: str = "data"
    upload_dir: str = "uploads"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    
    # Processing settings
    max_concurrent_jobs: int = 10
    job_timeout_minutes: int = 30
    
    # GPU settings
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Model settings
    models_dir: str = "models"
    default_model: str = "adaptive_fusion"
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_interval_seconds: int = 60
    
    # External API settings
    external_api_timeout: int = 30
    external_api_retries: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator("allowed_origins", pre=True)
    def validate_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_environments = ["development", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v.lower()


class DevelopmentSettings(Settings):
    """Development environment settings"""
    environment: str = "development"
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    """Production environment settings"""
    environment: str = "production"
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    allowed_origins: List[str] = ["https://yourfrontend.com"]


class TestingSettings(Settings):
    """Testing environment settings"""
    environment: str = "testing"
    debug: bool = True
    database_url: str = "sqlite:///./test.db"
    log_level: str = "DEBUG"


def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()