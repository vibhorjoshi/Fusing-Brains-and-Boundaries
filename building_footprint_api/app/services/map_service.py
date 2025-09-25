"""
Map Service for satellite imagery
"""

import aiohttp
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import base64
import asyncio
from PIL import Image
import io
import time
import math

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class MapService:
    """Service for fetching and managing satellite imagery"""
    
    def __init__(self):
        self.api_key = settings.MAPS_API_KEY
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour cache TTL
        
    async def get_satellite_image(
        self, 
        bounds: Dict[str, float], 
        zoom_level: int,
        width: int = 1280,
        height: int = 1280,
        maptype: str = "satellite"
    ) -> np.ndarray:
        """
        Fetch satellite image for the given bounds
        
        Args:
            bounds: Dictionary with 'north', 'south', 'east', 'west' coordinates
            zoom_level: Map zoom level (1-21)
            width: Image width in pixels
            height: Image height in pixels
            maptype: Map type (satellite, roadmap, hybrid, terrain)
            
        Returns:
            numpy.ndarray: Image data as numpy array
        """
        # Check cache first
        cache_key = f"{bounds}_{zoom_level}_{width}_{height}_{maptype}"
        cached = self.get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Calculate center
        lat = (bounds['north'] + bounds['south']) / 2
        lng = (bounds['east'] + bounds['west']) / 2
        
        # Build URL for Google Maps Static API
        url = f"https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lng}",
            "zoom": zoom_level,
            "size": f"{width}x{height}",
            "maptype": maptype,
            "key": self.api_key,
            "format": "png",
            "scale": 1
        }
        
        # Add bounds if specified
        if 'north' in bounds and 'south' in bounds and 'east' in bounds and 'west' in bounds:
            # Calculate visible area based on bounds
            visible = f"{bounds['south']},{bounds['west']}|{bounds['north']},{bounds['east']}"
            params["visible"] = visible
        
        # Fetch image
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    
                    # Convert to numpy array
                    image = Image.open(io.BytesIO(image_data))
                    image_array = np.array(image)
                    
                    # Store in cache
                    self.add_to_cache(cache_key, image_array)
                    
                    return image_array
                    
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching satellite image: {str(e)}")
                # Return black image as fallback
                return np.zeros((height, width, 3), dtype=np.uint8)
    
    async def get_tiles(
        self,
        bounds: Dict[str, float],
        zoom_level: int,
        maptype: str = "satellite"
    ) -> List[Dict[str, Any]]:
        """
        Get map tiles for the given bounds
        
        Args:
            bounds: Dictionary with 'north', 'south', 'east', 'west' coordinates
            zoom_level: Map zoom level (1-21)
            maptype: Map type (satellite, roadmap, hybrid, terrain)
            
        Returns:
            List of tile information including URLs
        """
        # Calculate tile coordinates
        north_west = self._latlon_to_tile(bounds['north'], bounds['west'], zoom_level)
        south_east = self._latlon_to_tile(bounds['south'], bounds['east'], zoom_level)
        
        tiles = []
        for x in range(north_west[0], south_east[0] + 1):
            for y in range(north_west[1], south_east[1] + 1):
                # Get tile URL
                url = self._get_tile_url(x, y, zoom_level, maptype)
                
                # Calculate lat/lon bounds for this tile
                tile_bounds = self._tile_to_latlon_bounds(x, y, zoom_level)
                
                tiles.append({
                    "x": x,
                    "y": y,
                    "z": zoom_level,
                    "url": url,
                    "bounds": tile_bounds
                })
        
        return tiles
    
    def _latlon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude/longitude to tile coordinates"""
        # Based on Slippy Map tilenames standard
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return (x, y)
    
    def _tile_to_latlon_bounds(self, x: int, y: int, zoom: int) -> Dict[str, float]:
        """Convert tile coordinates to latitude/longitude bounds"""
        n = 2.0 ** zoom
        lon1 = x / n * 360.0 - 180.0
        lon2 = (x + 1) / n * 360.0 - 180.0
        
        lat1_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat1 = math.degrees(lat1_rad)
        lat2_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
        lat2 = math.degrees(lat2_rad)
        
        return {
            "north": max(lat1, lat2),
            "south": min(lat1, lat2),
            "east": max(lon1, lon2),
            "west": min(lon1, lon2)
        }
    
    def _get_tile_url(self, x: int, y: int, zoom: int, maptype: str) -> str:
        """Get URL for a specific tile"""
        # Google Maps tile URL format
        return f"https://mt0.google.com/vt/lyrs={self._get_maptype_code(maptype)}&x={x}&y={y}&z={zoom}&key={self.api_key}"
    
    def _get_maptype_code(self, maptype: str) -> str:
        """Get map type code for tile URLs"""
        maptype_codes = {
            "roadmap": "m",
            "satellite": "s",
            "terrain": "t",
            "hybrid": "y"
        }
        return maptype_codes.get(maptype, "s")  # Default to satellite
    
    def get_from_cache(self, key: str) -> Any:
        """Get item from cache"""
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item["timestamp"] < self.cache_ttl:
                return item["data"]
            else:
                # Expired
                del self.cache[key]
        return None
    
    def add_to_cache(self, key: str, data: Any):
        """Add item to cache"""
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
        
        # Simple cache size management
        if len(self.cache) > 100:
            # Remove oldest items
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"]
            )[:10]
            for old_key in oldest_keys:
                del self.cache[old_key]