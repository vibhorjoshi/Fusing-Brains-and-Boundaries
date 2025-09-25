"""
Main API router - combines all endpoint modules
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    buildings,
    jobs,
    files,
    admin,
    ml_processing
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(buildings.router, prefix="/buildings", tags=["Building Footprints"])  
api_router.include_router(jobs.router, prefix="/jobs", tags=["Processing Jobs"])
api_router.include_router(files.router, prefix="/files", tags=["File Management"])
api_router.include_router(ml_processing.router, prefix="/ml", tags=["ML Processing"])
api_router.include_router(admin.router, prefix="/admin", tags=["Administration"])