"""
API dependencies module
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import get_settings
from app.core.security import get_current_user
from app.services.pipeline import PipelineService

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

# Dependency for PipelineService
pipeline_service = PipelineService()

def get_pipeline_service() -> PipelineService:
    """Get pipeline service instance"""
    return pipeline_service

# Optional auth dependency - uncomment to enforce authentication
# async def get_current_active_user(current_user = Depends(get_current_user)):
#     if not current_user:
#         raise HTTPException(status_code=401, detail="Inactive user")
#     return current_user