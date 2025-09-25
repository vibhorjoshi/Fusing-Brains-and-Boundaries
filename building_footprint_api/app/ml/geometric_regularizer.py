"""
Geometric Regularization Pipeline for Building Footprint Refinement
Advanced post-processing algorithms for polygon regularization and smoothing
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import cv2
from shapely.geometry import Polygon, Point, LinearRing, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
import shapely.geometry as sg
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
import logging

logger = logging.getLogger(__name__)


class GeometricRegularizer:
    """
    Advanced geometric regularization for building footprint polygons
    Implements corner detection, angle regularization, and shape simplification
    """
    
    def __init__(self, 
                 angle_tolerance: float = 15.0,
                 min_area: float = 25.0,
                 douglas_peucker_tolerance: float = 1.0,
                 corner_threshold: float = 0.7):
        """
        Initialize the geometric regularizer
        
        Args:
            angle_tolerance: Tolerance for angle regularization (degrees)
            min_area: Minimum area for valid polygons
            douglas_peucker_tolerance: Tolerance for Douglas-Peucker simplification
            corner_threshold: Threshold for corner detection
        """
        self.angle_tolerance = angle_tolerance
        self.min_area = min_area
        self.douglas_peucker_tolerance = douglas_peucker_tolerance
        self.corner_threshold = corner_threshold
        
    def regularize_polygon(self, polygon: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """
        Apply complete regularization pipeline to a polygon
        
        Args:
            polygon: List of (x, y) coordinate tuples
            
        Returns:
            Regularized polygon or None if invalid
        """
        try:
            # Convert to numpy array for processing
            coords = np.array(polygon)
            
            # Ensure polygon is closed
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])
                
            # Step 1: Douglas-Peucker simplification
            simplified = self._douglas_peucker_simplify(coords)
            
            # Step 2: Corner detection and refinement
            corners = self._detect_corners(simplified)
            
            # Step 3: Angle regularization (snap to 90-degree angles)
            regularized = self._regularize_angles(corners)
            
            # Step 4: Shape optimization
            optimized = self._optimize_shape(regularized)
            
            # Step 5: Final validation
            final_polygon = self._validate_polygon(optimized)
            
            return final_polygon
            
        except Exception as e:
            logger.error(f"Error in polygon regularization: {e}")
            return None
            
    def _douglas_peucker_simplify(self, coords: np.ndarray) -> np.ndarray:
        """Apply Douglas-Peucker algorithm for polygon simplification"""
        def perpendicular_distance(point, line_start, line_end):
            """Calculate perpendicular distance from point to line"""
            if np.allclose(line_start, line_end):
                return np.linalg.norm(point - line_start)
            
            # Vector from line_start to line_end
            line_vec = line_end - line_start
            # Vector from line_start to point
            point_vec = point - line_start
            
            # Project point onto line
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                return np.linalg.norm(point_vec)
                
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
            projection = line_start + t * line_vec
            
            return np.linalg.norm(point - projection)
        
        def douglas_peucker_recursive(coords, epsilon):
            """Recursive Douglas-Peucker implementation"""
            if len(coords) < 3:
                return coords
                
            # Find the point with maximum distance from line connecting first and last points
            max_dist = 0
            max_index = 0
            
            for i in range(1, len(coords) - 1):
                dist = perpendicular_distance(coords[i], coords[0], coords[-1])
                if dist > max_dist:
                    max_dist = dist
                    max_index = i
                    
            # If max distance is greater than epsilon, recursively simplify
            if max_dist > epsilon:
                # Recursive call on both parts
                left_result = douglas_peucker_recursive(coords[:max_index + 1], epsilon)
                right_result = douglas_peucker_recursive(coords[max_index:], epsilon)
                
                # Combine results (remove duplicate point)
                return np.vstack([left_result[:-1], right_result])
            else:
                # Return simplified line (just endpoints)
                return np.array([coords[0], coords[-1]])
        
        return douglas_peucker_recursive(coords, self.douglas_peucker_tolerance)
    
    def _detect_corners(self, coords: np.ndarray) -> np.ndarray:
        """Detect corners using curvature analysis"""
        if len(coords) < 4:
            return coords
            
        corners = [coords[0]]  # Always include first point
        
        for i in range(1, len(coords) - 1):
            # Calculate vectors to adjacent points
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            
            # Normalize vectors
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
            
            # Calculate angle between vectors
            dot_product = np.dot(v1_norm, v2_norm)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical errors
            angle = np.arccos(dot_product)
            
            # If angle is significant (not close to 180 degrees), it's a corner
            if angle > self.corner_threshold:
                corners.append(coords[i])
                
        corners.append(coords[-1])  # Always include last point
        
        return np.array(corners)
    
    def _regularize_angles(self, coords: np.ndarray) -> np.ndarray:
        """Regularize angles to common architectural angles (90°, 45°, etc.)"""
        if len(coords) < 4:
            return coords
            
        regularized = coords.copy()
        
        # Target angles in radians
        target_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        angle_tolerance_rad = np.deg2rad(self.angle_tolerance)
        
        for i in range(1, len(coords) - 1):
            # Calculate current angle
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            
            # Calculate angle
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            
            # Normalize angles to [0, 2π]
            angle1 = angle1 % (2 * np.pi)
            angle2 = angle2 % (2 * np.pi)
            
            # Find closest target angles
            for target in target_angles:
                if abs(angle1 - target) < angle_tolerance_rad:
                    # Adjust point to align with target angle
                    length = np.linalg.norm(v1)
                    new_v1 = length * np.array([np.cos(target), np.sin(target)])
                    regularized[i] = coords[i-1] + new_v1
                    break
                    
        return regularized
    
    def _optimize_shape(self, coords: np.ndarray) -> np.ndarray:
        """Optimize shape using energy minimization"""
        if len(coords) < 4:
            return coords
            
        def energy_function(flat_coords):
            """Energy function combining smoothness and regularity"""
            # Reshape flat coordinates back to 2D
            coords_2d = flat_coords.reshape(-1, 2)
            
            # Smoothness energy (minimize curvature)
            smoothness_energy = 0
            for i in range(1, len(coords_2d) - 1):
                curvature = self._calculate_curvature(coords_2d[i-1], coords_2d[i], coords_2d[i+1])
                smoothness_energy += curvature ** 2
                
            # Regularity energy (prefer right angles)
            regularity_energy = 0
            for i in range(1, len(coords_2d) - 1):
                angle = self._calculate_interior_angle(coords_2d[i-1], coords_2d[i], coords_2d[i+1])
                # Penalize deviation from 90 degrees
                angle_deviation = min(abs(angle - np.pi/2), abs(angle - 3*np.pi/2))
                regularity_energy += angle_deviation ** 2
                
            return smoothness_energy + 0.5 * regularity_energy
        
        try:
            # Flatten coordinates for optimization
            flat_coords = coords.flatten()
            
            # Optimize using L-BFGS-B
            result = minimize(energy_function, flat_coords, method='L-BFGS-B')
            
            if result.success:
                optimized_coords = result.x.reshape(-1, 2)
                return optimized_coords
            else:
                return coords
                
        except Exception as e:
            logger.warning(f"Shape optimization failed: {e}")
            return coords
    
    def _calculate_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate curvature at point p2"""
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Lengths
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 < 1e-8 or len2 < 1e-8:
            return 0.0
            
        # Cross product magnitude (2D)
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        
        # Curvature formula
        curvature = 2 * cross / (len1 * len2 * (len1 + len2))
        
        return curvature
    
    def _calculate_interior_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate interior angle at point p2"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Calculate angle
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        return angle
    
    def _validate_polygon(self, coords: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Validate and clean up the final polygon"""
        try:
            # Remove duplicate consecutive points
            unique_coords = [coords[0]]
            for i in range(1, len(coords)):
                if not np.allclose(coords[i], coords[i-1], atol=1e-6):
                    unique_coords.append(coords[i])
                    
            coords = np.array(unique_coords)
            
            # Ensure minimum number of points for a polygon
            if len(coords) < 3:
                return None
                
            # Close polygon if needed
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])
                
            # Check area
            polygon = Polygon(coords)
            if not polygon.is_valid or polygon.area < self.min_area:
                return None
                
            # Convert back to list of tuples
            return [(float(x), float(y)) for x, y in coords[:-1]]  # Remove duplicate closing point
            
        except Exception as e:
            logger.error(f"Polygon validation failed: {e}")
            return None


class BuildingShapeAnalyzer:
    """
    Analyze and classify building shapes for enhanced regularization
    """
    
    def __init__(self):
        self.shape_classes = {
            'rectangular': {'min_rectangularity': 0.8},
            'l_shaped': {'min_concavity': 0.3, 'max_rectangularity': 0.6},
            'u_shaped': {'min_concavity': 0.4, 'max_rectangularity': 0.5},
            'complex': {'max_rectangularity': 0.4}
        }
        
    def classify_shape(self, polygon: List[Tuple[float, float]]) -> str:
        """Classify building shape type"""
        try:
            # Calculate shape metrics
            rectangularity = self._calculate_rectangularity(polygon)
            concavity = self._calculate_concavity(polygon)
            
            # Classify based on metrics
            if rectangularity >= self.shape_classes['rectangular']['min_rectangularity']:
                return 'rectangular'
            elif (concavity >= self.shape_classes['l_shaped']['min_concavity'] and 
                  rectangularity <= self.shape_classes['l_shaped']['max_rectangularity']):
                return 'l_shaped'
            elif (concavity >= self.shape_classes['u_shaped']['min_concavity'] and 
                  rectangularity <= self.shape_classes['u_shaped']['max_rectangularity']):
                return 'u_shaped'
            else:
                return 'complex'
                
        except Exception as e:
            logger.error(f"Shape classification failed: {e}")
            return 'complex'
    
    def _calculate_rectangularity(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate how rectangular a polygon is (0-1)"""
        try:
            poly = Polygon(polygon)
            if not poly.is_valid:
                return 0.0
                
            # Get minimum bounding rectangle
            min_rect = poly.minimum_rotated_rectangle
            
            # Calculate rectangularity as ratio of areas
            rectangularity = poly.area / min_rect.area
            
            return min(1.0, rectangularity)
            
        except Exception:
            return 0.0
    
    def _calculate_concavity(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate concavity index (0-1)"""
        try:
            poly = Polygon(polygon)
            if not poly.is_valid:
                return 0.0
                
            # Get convex hull
            convex_hull = poly.convex_hull
            
            # Calculate concavity as area difference ratio
            concavity = (convex_hull.area - poly.area) / convex_hull.area
            
            return min(1.0, concavity)
            
        except Exception:
            return 0.0


class AdaptiveRegularizer:
    """
    Adaptive regularization that adjusts parameters based on building shape and size
    """
    
    def __init__(self):
        self.base_regularizer = GeometricRegularizer()
        self.shape_analyzer = BuildingShapeAnalyzer()
        
        # Shape-specific regularization parameters
        self.shape_params = {
            'rectangular': {
                'angle_tolerance': 10.0,
                'douglas_peucker_tolerance': 0.5,
                'corner_threshold': 0.5
            },
            'l_shaped': {
                'angle_tolerance': 15.0,
                'douglas_peucker_tolerance': 1.0,
                'corner_threshold': 0.6
            },
            'u_shaped': {
                'angle_tolerance': 15.0,
                'douglas_peucker_tolerance': 1.0,
                'corner_threshold': 0.6
            },
            'complex': {
                'angle_tolerance': 20.0,
                'douglas_peucker_tolerance': 1.5,
                'corner_threshold': 0.7
            }
        }
        
    def regularize_adaptive(self, polygon: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """Apply adaptive regularization based on shape classification"""
        try:
            # Classify shape
            shape_type = self.shape_analyzer.classify_shape(polygon)
            
            # Get shape-specific parameters
            params = self.shape_params.get(shape_type, self.shape_params['complex'])
            
            # Create regularizer with adapted parameters
            regularizer = GeometricRegularizer(
                angle_tolerance=params['angle_tolerance'],
                douglas_peucker_tolerance=params['douglas_peucker_tolerance'],
                corner_threshold=params['corner_threshold']
            )
            
            # Apply regularization
            regularized_polygon = regularizer.regularize_polygon(polygon)
            
            logger.info(f"Applied {shape_type} regularization")
            
            return regularized_polygon
            
        except Exception as e:
            logger.error(f"Adaptive regularization failed: {e}")
            return self.base_regularizer.regularize_polygon(polygon)


class BatchRegularizer:
    """
    Batch processing for multiple building polygons with progress tracking
    """
    
    def __init__(self, use_adaptive: bool = True):
        self.regularizer = AdaptiveRegularizer() if use_adaptive else GeometricRegularizer()
        
    def process_polygons(self, polygons: List[List[Tuple[float, float]]]) -> Dict:
        """Process multiple polygons with detailed results"""
        results = {
            'processed_count': 0,
            'successful_count': 0,
            'failed_count': 0,
            'regularized_polygons': [],
            'processing_stats': {}
        }
        
        for i, polygon in enumerate(polygons):
            try:
                if isinstance(self.regularizer, AdaptiveRegularizer):
                    regularized = self.regularizer.regularize_adaptive(polygon)
                else:
                    regularized = self.regularizer.regularize_polygon(polygon)
                    
                if regularized is not None:
                    results['regularized_polygons'].append(regularized)
                    results['successful_count'] += 1
                else:
                    results['regularized_polygons'].append(None)
                    results['failed_count'] += 1
                    
                results['processed_count'] += 1
                
            except Exception as e:
                logger.error(f"Error processing polygon {i}: {e}")
                results['regularized_polygons'].append(None)
                results['failed_count'] += 1
                results['processed_count'] += 1
                
        # Calculate statistics
        results['processing_stats'] = {
            'success_rate': results['successful_count'] / max(1, results['processed_count']),
            'failure_rate': results['failed_count'] / max(1, results['processed_count'])
        }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Create sample irregular polygon
    irregular_polygon = [
        (10.2, 10.1), (50.8, 12.3), (48.9, 45.7), (52.1, 48.2), 
        (12.4, 46.8), (9.7, 25.3), (10.2, 10.1)
    ]
    
    # Test regularization
    regularizer = AdaptiveRegularizer()
    regularized = regularizer.regularize_adaptive(irregular_polygon)
    
    print("Geometric Regularization Test Results:")
    print(f"Original polygon: {len(irregular_polygon)} vertices")
    if regularized:
        print(f"Regularized polygon: {len(regularized)} vertices")
        print(f"Regularized coordinates: {regularized}")
    else:
        print("Regularization failed")
        
    # Test batch processing
    batch_regularizer = BatchRegularizer(use_adaptive=True)
    test_polygons = [irregular_polygon] * 5
    
    batch_results = batch_regularizer.process_polygons(test_polygons)
    print(f"\nBatch processing results:")
    print(f"Success rate: {batch_results['processing_stats']['success_rate']:.2%}")
    print(f"Processed {batch_results['successful_count']}/{batch_results['processed_count']} polygons successfully")