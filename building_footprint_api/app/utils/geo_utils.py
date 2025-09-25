"""
Utility functions for the building footprint API
"""

import os
import uuid
import base64
import numpy as np
import cv2
import json
import logging
from io import BytesIO
from PIL import Image
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import geojson
from shapely.geometry import Polygon, mapping
import pyproj
from shapely.ops import transform
from functools import partial

logger = logging.getLogger(__name__)

def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode numpy array image to base64 string
    
    Args:
        image: Numpy array image
        
    Returns:
        Base64 encoded string
    """
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image")
    
    return base64.b64encode(encoded_image).decode('utf-8')

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to numpy array image
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        Numpy array image
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image)

def generate_unique_filename(prefix: str = "", suffix: str = "") -> str:
    """
    Generate a unique filename
    
    Args:
        prefix: Optional prefix for the filename
        suffix: Optional suffix for the filename
        
    Returns:
        Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    return f"{prefix}_{timestamp}_{unique_id}{suffix}" if prefix else f"{timestamp}_{unique_id}{suffix}"

def save_temp_file(data: bytes, extension: str = ".png") -> str:
    """
    Save binary data to a temporary file
    
    Args:
        data: Binary data to save
        extension: File extension
        
    Returns:
        Path to the saved file
    """
    from app.core.config import get_settings
    settings = get_settings()
    
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    filename = generate_unique_filename(suffix=extension)
    filepath = os.path.join(settings.TEMP_DIR, filename)
    
    with open(filepath, 'wb') as f:
        f.write(data)
    
    return filepath

def polygon_to_geojson(polygon: List[Tuple[float, float]], properties: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convert polygon points to GeoJSON
    
    Args:
        polygon: List of (x, y) coordinates defining the polygon
        properties: Optional properties for the GeoJSON feature
        
    Returns:
        GeoJSON feature
    """
    # Convert to Shapely polygon
    poly = Polygon(polygon)
    
    # Convert to GeoJSON Feature
    feature = {
        "type": "Feature",
        "geometry": mapping(poly),
        "properties": properties or {}
    }
    
    return feature

def reproject_polygon(polygon: List[Tuple[float, float]], src_crs: str, dst_crs: str) -> List[Tuple[float, float]]:
    """
    Reproject polygon coordinates from one CRS to another
    
    Args:
        polygon: List of (x, y) coordinates defining the polygon
        src_crs: Source coordinate reference system (e.g., 'EPSG:3857')
        dst_crs: Destination coordinate reference system (e.g., 'EPSG:4326')
        
    Returns:
        Reprojected polygon coordinates
    """
    # Create transformer
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    # Convert to shapely polygon
    poly = Polygon(polygon)
    
    # Apply transformation
    transformed_poly = transform(transformer.transform, poly)
    
    # Extract coordinates
    return list(transformed_poly.exterior.coords)

def pixel_to_geo_coords(
    pixel_x: float,
    pixel_y: float,
    bounds: Dict[str, float],
    image_width: int,
    image_height: int
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates
    
    Args:
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate
        bounds: Dictionary with 'north', 'south', 'east', 'west' coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        (longitude, latitude) tuple
    """
    # Calculate proportional position in the image
    x_ratio = pixel_x / image_width
    y_ratio = pixel_y / image_height
    
    # Calculate longitude and latitude
    lon = bounds['west'] + (bounds['east'] - bounds['west']) * x_ratio
    lat = bounds['north'] - (bounds['north'] - bounds['south']) * y_ratio
    
    return (lon, lat)

def geo_to_pixel_coords(
    lon: float,
    lat: float,
    bounds: Dict[str, float],
    image_width: int,
    image_height: int
) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates
    
    Args:
        lon: Longitude
        lat: Latitude
        bounds: Dictionary with 'north', 'south', 'east', 'west' coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        (pixel_x, pixel_y) tuple
    """
    # Calculate proportional position in the bounding box
    x_ratio = (lon - bounds['west']) / (bounds['east'] - bounds['west'])
    y_ratio = (bounds['north'] - lat) / (bounds['north'] - bounds['south'])
    
    # Calculate pixel coordinates
    pixel_x = int(x_ratio * image_width)
    pixel_y = int(y_ratio * image_height)
    
    return (pixel_x, pixel_y)

def mask_to_polygons(mask: np.ndarray, min_area: int = 100, epsilon: float = 1.0) -> List[List[Tuple[int, int]]]:
    """
    Convert binary mask to polygons
    
    Args:
        mask: Binary mask as numpy array
        min_area: Minimum contour area to keep
        epsilon: Approximation accuracy parameter for Douglas-Peucker algorithm
        
    Returns:
        List of polygons, each polygon is a list of (x, y) points
    """
    # Ensure mask is binary
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
        
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Filter small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Approximate contour to reduce number of points
        epsilon_value = epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_value, True)
        
        # Convert to list of tuples
        polygon = [(point[0][0], point[0][1]) for point in approx]
        
        # Ensure polygon has at least 3 points
        if len(polygon) >= 3:
            polygons.append(polygon)
            
    return polygons

def smooth_polygon(polygon: List[Tuple[int, int]], smoothing_factor: float = 0.2) -> List[Tuple[int, int]]:
    """
    Smooth polygon vertices
    
    Args:
        polygon: List of (x, y) coordinates defining the polygon
        smoothing_factor: Smoothing strength (0.0 - 1.0)
        
    Returns:
        Smoothed polygon
    """
    if len(polygon) <= 3:
        return polygon
    
    # Ensure polygon is closed (first point equals last point)
    closed = polygon[0] == polygon[-1]
    
    # If polygon is not closed, we need to handle differently
    working_polygon = polygon.copy()
    if not closed:
        # Add first point to the end to make it circular for smoothing
        working_polygon.append(working_polygon[0])
    
    smoothed = []
    for i in range(len(working_polygon)):
        # Get previous, current and next points
        prev_idx = (i - 1) % len(working_polygon)
        current_idx = i
        next_idx = (i + 1) % len(working_polygon)
        
        prev = working_polygon[prev_idx]
        current = working_polygon[current_idx]
        next_pt = working_polygon[next_idx]
        
        # Calculate smoothed point
        x = current[0] * (1 - smoothing_factor) + (prev[0] + next_pt[0]) * 0.5 * smoothing_factor
        y = current[1] * (1 - smoothing_factor) + (prev[1] + next_pt[1]) * 0.5 * smoothing_factor
        
        smoothed.append((int(x), int(y)))
    
    # If original polygon was not closed, remove last point
    if not closed:
        smoothed.pop()
    
    return smoothed

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score (0.0 - 1.0)
    """
    # Ensure masks are binary
    mask1_binary = mask1 > 0
    mask2_binary = mask2 > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return float(iou)