"""
Production Configuration Settings
AWS and Database Configuration for Production Deployment
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Settings
    APP_NAME: str = "Building Footprint AI Backend"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # Server Settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    API_V1_STR: str = "/api/v1"
    
    # Security Settings
    SECRET_KEY: str = Field(default="your-super-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Database Settings
    DATABASE_URL: str = Field(
        default="postgresql://username:password@localhost:5432/building_footprint_db",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_POOL_OVERFLOW: int = Field(default=20, env="DATABASE_POOL_OVERFLOW")
    
    # AWS Configuration
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # AWS S3 Settings
    AWS_S3_BUCKET: str = Field(default="building-footprint-ai-storage", env="AWS_S3_BUCKET")
    AWS_S3_PREFIX: str = Field(default="production/", env="AWS_S3_PREFIX")
    
    # AWS RDS Settings  
    AWS_RDS_ENDPOINT: Optional[str] = Field(default=None, env="AWS_RDS_ENDPOINT")
    AWS_RDS_DATABASE: str = Field(default="building_footprint_prod", env="AWS_RDS_DATABASE")
    
    # AWS SQS Settings
    AWS_SQS_QUEUE_URL: Optional[str] = Field(default=None, env="AWS_SQS_QUEUE_URL")
    AWS_SQS_DLQ_URL: Optional[str] = Field(default=None, env="AWS_SQS_DLQ_URL")
    
    # Redis Settings (for Celery)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # File Upload Settings
    MAX_FILE_SIZE_MB: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "image/tiff", "application/zip"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # ML Pipeline Settings
    ML_MODEL_PATH: str = Field(default="./models/", env="ML_MODEL_PATH")
    ML_CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="ML_CONFIDENCE_THRESHOLD")
    ML_BATCH_SIZE: int = Field(default=4, env="ML_BATCH_SIZE")
    ML_MAX_WORKERS: int = Field(default=2, env="ML_MAX_WORKERS")
    
    # Monitoring and Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # AWS CloudWatch
    CLOUDWATCH_LOG_GROUP: str = Field(default="/aws/ecs/building-footprint-ai", env="CLOUDWATCH_LOG_GROUP")
    CLOUDWATCH_LOG_STREAM: str = Field(default="backend", env="CLOUDWATCH_LOG_STREAM")
    
    @property
    def database_url_for_production(self) -> str:
        """Generate production database URL from AWS RDS settings"""
        if self.AWS_RDS_ENDPOINT and self.ENVIRONMENT == "production":
            # For production, construct URL from RDS settings
            return f"postgresql://user:password@{self.AWS_RDS_ENDPOINT}:5432/{self.AWS_RDS_DATABASE}"
        return self.DATABASE_URL
    
    @property
    def aws_s3_url(self) -> str:
        """Generate S3 bucket URL"""
        return f"https://{self.AWS_S3_BUCKET}.s3.{self.AWS_REGION}.amazonaws.com/"
    
    def get_aws_config(self) -> dict:
        """Get AWS configuration dictionary"""
        config = {
            "region_name": self.AWS_REGION
        }
        
        if self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY:
            config.update({
                "aws_access_key_id": self.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY
            })
        
        return config
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Environment-specific configurations
def get_environment_config():
    """Get environment-specific configuration"""
    if settings.ENVIRONMENT == "production":
        return {
            "debug": False,
            "testing": False,
            "database_url": settings.database_url_for_production,
            "cors_origins": ["https://yourdomain.com"],
            "log_level": "WARNING"
        }
    elif settings.ENVIRONMENT == "staging":
        return {
            "debug": False,
            "testing": True,
            "database_url": settings.DATABASE_URL,
            "cors_origins": ["https://staging.yourdomain.com"],
            "log_level": "INFO"
        }
    else:  # development
        return {
            "debug": True,
            "testing": False,
            "database_url": settings.DATABASE_URL,
            "cors_origins": settings.ALLOWED_ORIGINS,
            "log_level": "DEBUG"
        }

# AWS Services Configuration
AWS_SERVICES_CONFIG = {
    "s3": {
        "bucket": settings.AWS_S3_BUCKET,
        "prefix": settings.AWS_S3_PREFIX,
        "region": settings.AWS_REGION
    },
    "rds": {
        "endpoint": settings.AWS_RDS_ENDPOINT,
        "database": settings.AWS_RDS_DATABASE,
        "region": settings.AWS_REGION
    },
    "sqs": {
        "queue_url": settings.AWS_SQS_QUEUE_URL,
        "dlq_url": settings.AWS_SQS_DLQ_URL,
        "region": settings.AWS_REGION
    },
    "cloudwatch": {
        "log_group": settings.CLOUDWATCH_LOG_GROUP,
        "log_stream": settings.CLOUDWATCH_LOG_STREAM,
        "region": settings.AWS_REGION
    }
}