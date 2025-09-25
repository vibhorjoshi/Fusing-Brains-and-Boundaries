"""
Regularization Components for Building Footprints
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RansacThresholding:
    """RT: RANSAC-based Thresholding for robust line detection"""
    
    def __init__(self, canny_low: int = 50, canny_high: int = 150):
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Apply RANSAC-based thresholding"""
        try:
            # Edge detection
            edges = cv2.Canny(
                (mask * 255).astype(np.uint8), 
                self.canny_low, 
                self.canny_high
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return mask.astype(np.float32)
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate polygon using Douglas-Peucker algorithm
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Create regularized mask
            regularized_mask = np.zeros_like(mask)
            cv2.fillPoly(regularized_mask, [np.array(approx, dtype=np.int32)], 1)
            
            return regularized_mask.astype(np.float32)
            
        except Exception as e:
            logger.error(f"RT regularization failed: {str(e)}")
            return mask.astype(np.float32)

class RectangularRegularization:
    """RR: Rectangular Regularization for orthogonality enforcement"""
    
    def __init__(self):
        pass
    
    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Apply rectangular regularization"""
        try:
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8), 
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return mask.astype(np.float32)
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get rotated rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Create regularized mask with rectangular shape
            regularized_mask = np.zeros_like(mask)
            cv2.fillPoly(regularized_mask, [np.array(box, dtype=np.int32)], 1)
            
            return regularized_mask.astype(np.float32)
            
        except Exception as e:
            logger.error(f"RR regularization failed: {str(e)}")
            return mask.astype(np.float32)

class FeatureEnhancementRegularization:
    """FER: Feature Enhancement Regularization for edge preservation"""
    
    def __init__(self, canny_low: int = 50, canny_high: int = 150):
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Apply feature enhancement regularization"""
        try:
            # Edge detection
            edges = cv2.Canny(
                (mask * 255).astype(np.uint8),
                self.canny_low,
                self.canny_high
            )
            
            # Dilate edges
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Combine edges with mask
            enhanced_mask = np.logical_or(
                mask.astype(bool),
                dilated_edges.astype(bool)
            ).astype(np.float32)
            
            # Remove noise by closing operation
            kernel = np.ones((5, 5), np.uint8)
            enhanced_mask = cv2.morphologyEx(
                enhanced_mask, 
                cv2.MORPH_CLOSE, 
                kernel
            )
            
            return enhanced_mask
            
        except Exception as e:
            logger.error(f"FER regularization failed: {str(e)}")
            return mask.astype(np.float32)

class HybridRegularizer:
    """Combined regularization techniques"""
    
    def __init__(self):
        self.rt_regularizer = RansacThresholding()
        self.rr_regularizer = RectangularRegularization()
        self.fer_regularizer = FeatureEnhancementRegularization()
    
    def apply_rt(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """Apply RT regularization to all masks"""
        return [self.rt_regularizer.apply(mask) for mask in masks]
    
    def apply_rr(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """Apply RR regularization to all masks"""
        return [self.rr_regularizer.apply(mask) for mask in masks]
    
    def apply_fer(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """Apply FER regularization to all masks"""
        return [self.fer_regularizer.apply(mask) for mask in masks]
    
    def apply(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply all regularization techniques to a single mask
        Returns dictionary with all variants
        """
        return {
            "original": mask.copy(),
            "rt": self.rt_regularizer.apply(mask),
            "rr": self.rr_regularizer.apply(mask),
            "fer": self.fer_regularizer.apply(mask)
        }
    
    def apply_all(self, masks: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """Apply all regularization techniques to all masks"""
        return [self.apply(mask) for mask in masks]