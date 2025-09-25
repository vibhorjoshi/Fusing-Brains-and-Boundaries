"""
Test utility functions
"""

import pytest
import numpy as np
import base64
import io
from PIL import Image
import os

from app.utils.geo_utils import (
    encode_image_to_base64,
    decode_base64_to_image,
    polygon_to_geojson,
    mask_to_polygons,
    calculate_iou
)

def test_encode_decode_image():
    """Test image encoding/decoding to base64"""
    # Create a simple test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a white rectangle
    image[25:75, 25:75] = 255
    
    # Encode to base64
    base64_string = encode_image_to_base64(image)
    assert isinstance(base64_string, str)
    assert len(base64_string) > 0
    
    # Decode from base64
    decoded_image = decode_base64_to_image(base64_string)
    assert decoded_image.shape == image.shape
    assert np.array_equal(decoded_image, image)

def test_polygon_to_geojson():
    """Test converting polygon to GeoJSON"""
    # Create a simple polygon
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
    
    # Convert to GeoJSON
    geojson = polygon_to_geojson(polygon)
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "Polygon"
    assert len(geojson["geometry"]["coordinates"][0]) == len(polygon)
    
    # With properties
    properties = {"id": 1, "area": 1.0}
    geojson = polygon_to_geojson(polygon, properties)
    assert geojson["properties"] == properties

def test_mask_to_polygons():
    """Test converting mask to polygons"""
    # Create a simple mask with a square
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    
    # Convert to polygons
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1
    assert len(polygons[0]) >= 4  # At least 4 points for a square
    
    # Test with small contours that should be filtered
    small_mask = np.zeros((100, 100), dtype=np.uint8)
    small_mask[10:15, 10:15] = 255  # Small square
    small_mask[25:75, 25:75] = 255  # Larger square
    
    polygons = mask_to_polygons(small_mask, min_area=100)
    assert len(polygons) == 1  # Only the larger square should remain

def test_calculate_iou():
    """Test IoU calculation"""
    # Create two overlapping masks
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[25:75, 25:75] = 1
    
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[50:100, 50:100] = 1
    
    # Calculate IoU
    iou = calculate_iou(mask1, mask2)
    
    # Expected IoU: Intersection (25x25) / Union (75x75)
    expected_iou = (25 * 25) / (50 * 50 + 50 * 50 - 25 * 25)
    assert abs(iou - expected_iou) < 1e-6
    
    # Test with no overlap
    mask3 = np.zeros((100, 100), dtype=np.uint8)
    mask3[0:25, 0:25] = 1
    
    iou = calculate_iou(mask1, mask3)
    assert iou == 0.0
    
    # Test with identical masks
    iou = calculate_iou(mask1, mask1)
    assert iou == 1.0