"""
Maps API endpoints for Google Maps integration
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
import logging

from app.services.maps_service import GoogleMapsService
from app.models.schemas import BoundingBox

router = APIRouter()
logger = logging.getLogger(__name__)

maps_service = GoogleMapsService()

@router.get("/satellite-image")
async def get_satellite_image(
    bounds: BoundingBox,
    zoom: int = Query(default=18, ge=10, le=20),
    size: str = Query(default="1024x1024", regex=r"^\d+x\d+$")
):
    """
    Get satellite imagery for specified bounds
    """
    try:
        image_url = await maps_service.get_satellite_image(bounds, zoom, size)
        return {"image_url": image_url}
        
    except Exception as e:
        logger.error(f"Failed to get satellite image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/geocode")
async def geocode_address(address: str):
    """
    Convert address to coordinates
    """
    try:
        coordinates = await maps_service.geocode_address(address)
        return coordinates
        
    except Exception as e:
        logger.error(f"Geocoding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reverse-geocode")
async def reverse_geocode(lat: float, lng: float):
    """
    Convert coordinates to address
    """
    try:
        address = await maps_service.reverse_geocode(lat, lng)
        return address
        
    except Exception as e:
        logger.error(f"Reverse geocoding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))