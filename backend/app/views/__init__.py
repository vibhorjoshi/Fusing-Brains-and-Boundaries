"""
Views package for API routes
"""

from fastapi import APIRouter

# Create main router
api_router = APIRouter()

# Import route modules here when created
# from .auth_routes import router as auth_router
# from .processing_routes import router as processing_router

# api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
# api_router.include_router(processing_router, prefix="/processing", tags=["processing"])

__all__ = ["APIRouter"]