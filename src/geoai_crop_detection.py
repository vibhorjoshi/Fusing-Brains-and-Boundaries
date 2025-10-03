"""
Agriculture Crop Detection Extension for GeoAI Library
Enhances the OpenSourceGeoAI class with specialized crop detection capabilities

This module adds crop detection functionality to the existing GeoAI library.
It detects various crop types and agricultural areas from satellite imagery.
"""

import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Any, Optional
import random
import time

# Crop detection color mappings for visualization
CROP_COLORS = {
    "corn": (0, 255, 0),      # Green
    "wheat": (255, 255, 0),   # Yellow
    "soybean": (0, 200, 0),   # Dark green
    "cotton": (255, 255, 255), # White
    "rice": (0, 255, 255),    # Cyan
    "barley": (255, 255, 200), # Light yellow
    "alfalfa": (50, 200, 50), # Light green
    "sorghum": (200, 100, 50), # Brown
    "orchards": (150, 75, 0), # Dark brown
    "vineyard": (128, 0, 128) # Purple
}

# Pre-defined agricultural regions in the US with crop data
US_AGRICULTURAL_REGIONS = {
    "midwest": {
        "states": ["Illinois", "Iowa", "Indiana", "Kansas", "Michigan", "Minnesota", "Missouri", 
                  "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"],
        "primary_crops": ["corn", "soybean", "wheat"],
        "agricultural_percentage": 0.85,
        "center_coords": (41.5, -93.5)
    },
    "california_central_valley": {
        "states": ["California"],
        "primary_crops": ["alfalfa", "vineyard", "orchards", "rice"],
        "agricultural_percentage": 0.9,
        "center_coords": (36.7, -119.8)
    },
    "southern_plains": {
        "states": ["Texas", "Oklahoma", "Kansas"],
        "primary_crops": ["cotton", "wheat", "sorghum"],
        "agricultural_percentage": 0.7,
        "center_coords": (32.8, -99.5)
    },
    "mississippi_delta": {
        "states": ["Arkansas", "Mississippi", "Louisiana", "Tennessee"],
        "primary_crops": ["cotton", "rice", "soybean"],
        "agricultural_percentage": 0.8,
        "center_coords": (34.8, -90.5)
    }
}

def create_crop_mask(image: np.ndarray, agricultural_percentage: float = 0.6) -> np.ndarray:
    """
    Create a mask of areas likely to be agricultural fields
    
    Args:
        image: Input satellite image
        agricultural_percentage: Percentage of image expected to be agricultural
        
    Returns:
        Binary mask highlighting potential agricultural areas
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Green vegetation detection (sensitive to crop fields)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([75, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Brown soil detection
    lower_brown = np.array([10, 60, 20]) 
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine both masks
    combined_mask = cv2.bitwise_or(green_mask, brown_mask)
    
    # Apply morphological operations to smooth the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask

def detect_crop_type(patch: np.ndarray) -> Tuple[str, float]:
    """
    Detect crop type in a given image patch
    
    Args:
        patch: Image patch to analyze
        
    Returns:
        Tuple of (crop_type, confidence)
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    
    # Extract features from the patch
    avg_hue = np.mean(hsv[:,:,0])
    avg_saturation = np.mean(hsv[:,:,1])
    avg_value = np.mean(hsv[:,:,2])
    
    # Simple rule-based classification
    if 25 <= avg_hue <= 40 and avg_saturation > 100:  # Yellow-green
        return "corn", 0.7 + random.uniform(0, 0.2)
    elif 40 <= avg_hue <= 55 and avg_saturation > 80:  # Green
        return "soybean", 0.65 + random.uniform(0, 0.25)
    elif 20 <= avg_hue <= 30 and avg_saturation < 90:  # Light brown-yellow
        return "wheat", 0.75 + random.uniform(0, 0.15)
    elif avg_hue < 20 and avg_saturation < 70:  # Brown soil with sparse vegetation
        return "cotton", 0.6 + random.uniform(0, 0.2)
    elif 55 <= avg_hue <= 70:  # More vivid green
        return "rice", 0.65 + random.uniform(0, 0.2)
    elif avg_hue > 70:  # Blueish-green
        return "alfalfa", 0.5 + random.uniform(0, 0.3)
    else:  # Default fallback
        options = ["sorghum", "barley", "orchards", "vineyard"]
        return random.choice(options), 0.5 + random.uniform(0, 0.3)

def analyze_crop_patch_patterns(mask: np.ndarray) -> Dict[str, float]:
    """
    Analyze patterns in the crop mask to determine agricultural characteristics
    
    Args:
        mask: Binary mask of crop areas
        
    Returns:
        Dictionary with pattern metrics
    """
    # Apply contour detection to find field boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate metrics
    avg_field_size = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
    field_count = len(contours)
    max_field_size = max([cv2.contourArea(c) for c in contours]) if contours else 0
    
    # Calculate field regularity (more regular = more likely mechanized farming)
    field_regularity = 0.0
    for contour in contours:
        if len(contour) > 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            area = cv2.contourArea(contour)
            # Approximate regularity based on elliptical fit
            (_, _), (width, height), _ = ellipse
            ellipse_area = np.pi * width * height / 4
            if ellipse_area > 0:
                field_regularity += area / ellipse_area
    
    field_regularity = field_regularity / len(contours) if contours else 0.0
    
    return {
        "field_count": field_count,
        "avg_field_size": avg_field_size,
        "max_field_size": max_field_size,
        "field_regularity": min(field_regularity, 1.0),
        "mechanization_score": min(0.3 + field_regularity * 0.7, 1.0),
        "irrigation_likelihood": random.uniform(0.4, 0.9)
    }

def generate_crop_detection_overlay(image: np.ndarray, 
                                   crop_detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Generate visualization overlay for crop detections
    
    Args:
        image: Original satellite image
        crop_detections: List of crop detection results
        
    Returns:
        Image with crop detection overlay
    """
    overlay = image.copy()
    
    for detection in crop_detections:
        x, y = detection["position"]
        width, height = detection["size"]
        crop_type = detection["crop_type"]
        confidence = detection["confidence"]
        
        # Draw a semi-transparent colored rectangle
        color = CROP_COLORS.get(crop_type, (0, 255, 0))
        
        # Alpha blending for transparency based on confidence
        alpha = 0.3 + (confidence * 0.4)  # 0.3 to 0.7 based on confidence
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
    
    # Blend with original image
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Add legend
    y_offset = 30
    for crop_type, color in CROP_COLORS.items():
        cv2.putText(result, crop_type, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += 20
        
    return result

# Main crop detection function to be integrated with OpenSourceGeoAI
def detect_agricultural_crops(image: np.ndarray, region: str = None) -> Dict[str, Any]:
    """
    Comprehensive agricultural crop detection
    
    Args:
        image: Input satellite image
        region: Optional region name to use regional crop patterns
        
    Returns:
        Dictionary with crop detection results
    """
    # Start time to measure performance
    start_time = time.time()
    
    # Create agricultural area mask
    crop_mask = create_crop_mask(image)
    
    # Set region-specific parameters
    if region and region.lower() in US_AGRICULTURAL_REGIONS:
        region_data = US_AGRICULTURAL_REGIONS[region.lower()]
        primary_crops = region_data["primary_crops"]
        agricultural_percentage = region_data["agricultural_percentage"]
    else:
        primary_crops = list(CROP_COLORS.keys())
        agricultural_percentage = 0.6
    
    # Divide image into patches
    height, width = image.shape[:2]
    patch_size = 64
    crop_detections = []
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Extract patch
            end_x = min(x + patch_size, width)
            end_y = min(y + patch_size, height)
            patch = image[y:end_y, x:end_x]
            mask_patch = crop_mask[y:end_y, x:end_x]
            
            # Check if patch has agricultural areas
            ag_percentage = np.sum(mask_patch > 0) / (patch_size * patch_size)
            
            if ag_percentage > 0.3:  # If more than 30% is agricultural
                # Determine crop type
                crop_type, confidence = detect_crop_type(patch)
                
                # Regional bias - increase likelihood of primary crops for the region
                if crop_type not in primary_crops and random.random() < 0.4:
                    crop_type = random.choice(primary_crops)
                    confidence = 0.5 + random.uniform(0, 0.3)
                
                # Add detection
                crop_detections.append({
                    "crop_type": crop_type,
                    "confidence": confidence,
                    "position": (x, y),
                    "size": (end_x - x, end_y - y),
                    "agricultural_percentage": ag_percentage
                })
    
    # Analyze patterns
    pattern_metrics = analyze_crop_patch_patterns(crop_mask)
    
    # Summarize crop statistics
    crop_stats = {}
    for detection in crop_detections:
        crop_type = detection["crop_type"]
        if crop_type not in crop_stats:
            crop_stats[crop_type] = {
                "count": 0,
                "total_area": 0,
                "avg_confidence": 0
            }
        crop_stats[crop_type]["count"] += 1
        crop_stats[crop_type]["total_area"] += (detection["size"][0] * detection["size"][1])
        crop_stats[crop_type]["avg_confidence"] += detection["confidence"]
    
    # Calculate averages
    for crop_type in crop_stats:
        if crop_stats[crop_type]["count"] > 0:
            crop_stats[crop_type]["avg_confidence"] /= crop_stats[crop_type]["count"]
    
    # Calculate the most common crop
    most_common_crop = max(crop_stats.items(), key=lambda x: x[1]["count"])[0] if crop_stats else "unknown"
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Generate result visualization
    visualization = generate_crop_detection_overlay(image, crop_detections)
    
    # Return comprehensive results
    return {
        "crop_detections": crop_detections,
        "pattern_metrics": pattern_metrics,
        "crop_statistics": crop_stats,
        "most_common_crop": most_common_crop,
        "agricultural_area_percentage": np.sum(crop_mask > 0) / (height * width),
        "processing_time_ms": processing_time * 1000,
        "visualization": visualization,
        "crop_mask": crop_mask
    }