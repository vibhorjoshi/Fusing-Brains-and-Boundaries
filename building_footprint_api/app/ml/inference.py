"""
Common inference utilities
"""

import numpy as np
import torch
import cv2
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for model inference"""
    # Ensure image is in the right format (H, W, C) and normalize
    if len(image.shape) == 2:
        # Add channel dimension if grayscale
        image = np.expand_dims(image, axis=-1)
    
    if image.shape[-1] == 1:
        # Convert grayscale to RGB by repeating channels
        image = np.repeat(image, 3, axis=-1)
    
    # Convert to float32 and normalize to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensor and add batch dimension
    # Move from HWC to CHW format
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)

def extract_buildings_from_masks(masks: List[np.ndarray], confidence_scores: List[float] = None) -> List[Dict[str, Any]]:
    """Extract building polygons from binary masks"""
    buildings = []
    
    for i, mask in enumerate(masks):
        # Ensure mask is binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            continue
            
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Skip tiny contours
        if area < 10:
            continue
            
        # Create simplified polygon
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to GeoJSON format
        polygon = []
        for point in approx:
            x, y = point[0]
            polygon.append([float(x), float(y)])
        
        # Close the polygon
        polygon.append(polygon[0])
        
        # Create building object
        confidence = confidence_scores[i] if confidence_scores else 0.9
        building = {
            "id": f"building_{i}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            },
            "properties": {
                "confidence": float(confidence),
                "method": "hybrid_geoai"
            },
            "area": float(area),
            "mask": binary_mask
        }
        
        buildings.append(building)
    
    return buildings

def calculate_metrics(predicted_masks: List[np.ndarray], ground_truth_masks: List[np.ndarray]) -> Dict[str, float]:
    """Calculate evaluation metrics between predicted and ground truth masks"""
    metrics = {
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    
    if not predicted_masks or not ground_truth_masks:
        return metrics
    
    # Match predictions to ground truth by maximum IoU
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    # For each ground truth, find best matching prediction
    for gt_mask in ground_truth_masks:
        best_iou = 0.0
        best_pred = None
        
        for pred_mask in predicted_masks:
            # Calculate IoU
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = intersection / (union + 1e-6)
            
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_mask
        
        if best_pred is not None:
            # Calculate precision and recall
            tp = np.logical_and(gt_mask, best_pred).sum()
            fp = np.logical_and(np.logical_not(gt_mask), best_pred).sum()
            fn = np.logical_and(gt_mask, np.logical_not(best_pred)).sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            total_iou += best_iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1
    
    # Calculate average metrics
    num_gt = len(ground_truth_masks)
    metrics["iou"] = total_iou / num_gt if num_gt > 0 else 0.0
    metrics["precision"] = total_precision / num_gt if num_gt > 0 else 0.0
    metrics["recall"] = total_recall / num_gt if num_gt > 0 else 0.0
    metrics["f1"] = total_f1 / num_gt if num_gt > 0 else 0.0
    
    return metrics