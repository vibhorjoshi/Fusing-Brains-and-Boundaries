"""
Comprehensive evaluation metrics for building footprint extraction
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import math

class BuildingEvaluationMetrics:
    """Comprehensive evaluation metrics for building footprint extraction"""
    
    def __init__(self):
        """Initialize evaluation metrics calculator"""
        self.metrics_history = []
        
    def calculate_iou(self, pred_polygon: Polygon, gt_polygon: Polygon) -> float:
        """Calculate Intersection over Union for two polygons"""
        try:
            if not pred_polygon.is_valid:
                pred_polygon = pred_polygon.buffer(0)
            if not gt_polygon.is_valid:
                gt_polygon = gt_polygon.buffer(0)
                
            intersection = pred_polygon.intersection(gt_polygon)
            union = pred_polygon.union(gt_polygon)
            
            if union.area == 0:
                return 0.0
                
            return intersection.area / union.area
            
        except Exception as e:
            print(f"IoU calculation error: {e}")
            return 0.0
    
    def calculate_precision_recall(self, 
                                 predicted_polygons: List[Polygon],
                                 ground_truth_polygons: List[Polygon],
                                 iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        
        if not predicted_polygons:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
            
        if not ground_truth_polygons:
            return {"precision": 0.0, "recall": 1.0, "f1_score": 0.0}
        
        # Match predictions to ground truth
        tp = 0  # True positives
        matched_gt = set()
        
        for pred_poly in predicted_polygons:
            best_iou = 0
            best_match = -1
            
            for i, gt_poly in enumerate(ground_truth_polygons):
                if i in matched_gt:
                    continue
                    
                iou = self.calculate_iou(pred_poly, gt_poly)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = i
            
            if best_match != -1:
                tp += 1
                matched_gt.add(best_match)
        
        fp = len(predicted_polygons) - tp  # False positives
        fn = len(ground_truth_polygons) - tp  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    def calculate_geometric_accuracy(self, 
                                   predicted_polygons: List[Polygon],
                                   ground_truth_polygons: List[Polygon]) -> Dict[str, float]:
        """Calculate geometric accuracy metrics"""
        
        if not predicted_polygons or not ground_truth_polygons:
            return {"geometric_accuracy": 0.0, "area_accuracy": 0.0, "hausdorff_distance": float('inf')}
        
        total_area_error = 0
        total_hausdorff = 0
        valid_matches = 0
        
        for pred_poly in predicted_polygons:
            best_iou = 0
            best_gt = None
            
            for gt_poly in ground_truth_polygons:
                iou = self.calculate_iou(pred_poly, gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_poly
            
            if best_gt and best_iou > 0.3:  # Match threshold
                # Area accuracy
                area_error = abs(pred_poly.area - best_gt.area) / best_gt.area
                total_area_error += area_error
                
                # Hausdorff distance (simplified)
                hausdorff = self.calculate_hausdorff_distance(pred_poly, best_gt)
                total_hausdorff += hausdorff
                
                valid_matches += 1
        
        if valid_matches == 0:
            return {"geometric_accuracy": 0.0, "area_accuracy": 0.0, "hausdorff_distance": float('inf')}
        
        avg_area_error = total_area_error / valid_matches
        avg_hausdorff = total_hausdorff / valid_matches
        geometric_accuracy = 1.0 - min(avg_area_error, 1.0)  # Normalized
        
        return {
            "geometric_accuracy": geometric_accuracy,
            "area_accuracy": 1.0 - avg_area_error,
            "hausdorff_distance": avg_hausdorff
        }
    
    def calculate_hausdorff_distance(self, poly1: Polygon, poly2: Polygon) -> float:
        """Calculate simplified Hausdorff distance between two polygons"""
        try:
            # Sample points from polygon boundaries
            coords1 = list(poly1.exterior.coords)
            coords2 = list(poly2.exterior.coords)
            
            # Calculate max distance from poly1 to poly2
            max_dist_1_to_2 = 0
            for p1 in coords1:
                min_dist = min([Point(p1).distance(Point(p2)) for p2 in coords2])
                max_dist_1_to_2 = max(max_dist_1_to_2, min_dist)
            
            # Calculate max distance from poly2 to poly1
            max_dist_2_to_1 = 0
            for p2 in coords2:
                min_dist = min([Point(p2).distance(Point(p1)) for p1 in coords1])
                max_dist_2_to_1 = max(max_dist_2_to_1, min_dist)
            
            return max(max_dist_1_to_2, max_dist_2_to_1)
            
        except Exception:
            return float('inf')
    
    def calculate_regularization_metrics(self, 
                                       original_polygons: List[Polygon],
                                       regularized_polygons: List[Polygon]) -> Dict[str, float]:
        """Calculate metrics for geometric regularization effectiveness"""
        
        if len(original_polygons) != len(regularized_polygons):
            return {"regularization_error": 1.0}
        
        total_improvement = 0
        shape_preservation = 0
        corner_improvement = 0
        
        for orig, reg in zip(original_polygons, regularized_polygons):
            if not orig or not reg:
                continue
                
            # Shape preservation (IoU between original and regularized)
            preservation = self.calculate_iou(orig, reg)
            shape_preservation += preservation
            
            # Corner quality improvement
            orig_corners = self.count_near_right_angles(orig)
            reg_corners = self.count_near_right_angles(reg)
            
            corner_improvement += (reg_corners - orig_corners) / max(len(list(orig.exterior.coords)), 1)
            
            # Overall geometric improvement
            orig_regularity = self.calculate_polygon_regularity(orig)
            reg_regularity = self.calculate_polygon_regularity(reg)
            
            total_improvement += (reg_regularity - orig_regularity)
        
        n_polygons = len(original_polygons)
        
        return {
            "shape_preservation": shape_preservation / n_polygons,
            "corner_improvement": corner_improvement / n_polygons,
            "regularity_improvement": total_improvement / n_polygons,
            "regularization_success": (shape_preservation / n_polygons) > 0.8
        }
    
    def count_near_right_angles(self, polygon: Polygon, tolerance: float = 15.0) -> int:
        """Count corners that are close to right angles (90 degrees)"""
        coords = list(polygon.exterior.coords)[:-1]  # Exclude duplicate last point
        if len(coords) < 3:
            return 0
        
        right_angles = 0
        for i in range(len(coords)):
            p1 = coords[i-1]
            p2 = coords[i]
            p3 = coords[(i+1) % len(coords)]
            
            # Calculate angle
            angle = self.calculate_angle(p1, p2, p3)
            angle_deg = math.degrees(angle)
            
            # Check if close to 90 degrees
            if abs(angle_deg - 90) <= tolerance or abs(angle_deg - 270) <= tolerance:
                right_angles += 1
                
        return right_angles
    
    def calculate_angle(self, p1: Tuple[float, float], 
                       p2: Tuple[float, float], 
                       p3: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0
        
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        
        return math.acos(cos_angle)
    
    def calculate_polygon_regularity(self, polygon: Polygon) -> float:
        """Calculate how regular/rectangular a polygon is"""
        coords = list(polygon.exterior.coords)[:-1]
        if len(coords) < 4:
            return 0.0
        
        # Calculate side length variance
        side_lengths = []
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i+1) % len(coords)]
            length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            side_lengths.append(length)
        
        if len(side_lengths) == 0:
            return 0.0
        
        # Regularity based on side length consistency
        mean_length = np.mean(side_lengths)
        length_variance = np.var(side_lengths) / (mean_length**2) if mean_length > 0 else 1.0
        length_regularity = 1.0 / (1.0 + length_variance)
        
        # Regularity based on right angles
        right_angle_count = self.count_near_right_angles(polygon)
        angle_regularity = right_angle_count / len(coords)
        
        return (length_regularity + angle_regularity) / 2.0
    
    def generate_comprehensive_report(self,
                                    predicted_polygons: List[Polygon],
                                    ground_truth_polygons: List[Polygon],
                                    regularized_polygons: Optional[List[Polygon]] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        # Basic detection metrics
        detection_metrics = self.calculate_precision_recall(predicted_polygons, ground_truth_polygons)
        
        # Geometric accuracy
        geometric_metrics = self.calculate_geometric_accuracy(predicted_polygons, ground_truth_polygons)
        
        # IoU distribution
        iou_scores = []
        for pred in predicted_polygons:
            best_iou = 0
            for gt in ground_truth_polygons:
                iou = self.calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
            iou_scores.append(best_iou)
        
        # Regularization metrics if available
        regularization_metrics = {}
        if regularized_polygons:
            regularization_metrics = self.calculate_regularization_metrics(
                predicted_polygons, regularized_polygons
            )
        
        # Comprehensive report
        report = {
            "detection_performance": detection_metrics,
            "geometric_accuracy": geometric_metrics,
            "iou_statistics": {
                "mean_iou": np.mean(iou_scores) if iou_scores else 0.0,
                "median_iou": np.median(iou_scores) if iou_scores else 0.0,
                "std_iou": np.std(iou_scores) if iou_scores else 0.0,
                "min_iou": np.min(iou_scores) if iou_scores else 0.0,
                "max_iou": np.max(iou_scores) if iou_scores else 0.0
            },
            "dataset_statistics": {
                "predicted_buildings": len(predicted_polygons),
                "ground_truth_buildings": len(ground_truth_polygons),
                "detection_rate": len(predicted_polygons) / max(len(ground_truth_polygons), 1)
            },
            "regularization_performance": regularization_metrics,
            "overall_score": self.calculate_overall_score(detection_metrics, geometric_metrics)
        }
        
        # Store in history
        self.metrics_history.append(report)
        
        return report
    
    def calculate_overall_score(self, 
                              detection_metrics: Dict[str, float],
                              geometric_metrics: Dict[str, float]) -> float:
        """Calculate weighted overall performance score"""
        
        # Weights for different aspects
        weights = {
            "f1_score": 0.4,
            "geometric_accuracy": 0.3,
            "precision": 0.2,
            "recall": 0.1
        }
        
        score = (
            weights["f1_score"] * detection_metrics.get("f1_score", 0.0) +
            weights["geometric_accuracy"] * geometric_metrics.get("geometric_accuracy", 0.0) +
            weights["precision"] * detection_metrics.get("precision", 0.0) +
            weights["recall"] * detection_metrics.get("recall", 0.0)
        )
        
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]


class PerformanceProfiler:
    """Performance profiling for ML pipeline components"""
    
    def __init__(self):
        self.timing_data = {}
        
    def time_operation(self, operation_name: str, func, *args, **kwargs):
        """Time an operation and store results"""
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        
        if operation_name not in self.timing_data:
            self.timing_data[operation_name] = []
        
        self.timing_data[operation_name].append(elapsed_ms)
        
        return result
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all timed operations"""
        summary = {}
        
        for operation, times in self.timing_data.items():
            if times:
                summary[operation] = {
                    "mean_time_ms": np.mean(times),
                    "median_time_ms": np.median(times),
                    "std_time_ms": np.std(times),
                    "min_time_ms": np.min(times),
                    "max_time_ms": np.max(times),
                    "total_calls": len(times)
                }
        
        return summary


# Usage example functions
def evaluate_building_extraction_pipeline(predictions: List[Polygon],
                                        ground_truth: List[Polygon],
                                        regularized: Optional[List[Polygon]] = None) -> Dict[str, Any]:
    """Convenience function for complete pipeline evaluation"""
    
    evaluator = BuildingEvaluationMetrics()
    return evaluator.generate_comprehensive_report(predictions, ground_truth, regularized)


def profile_performance(mask_rcnn_func, regularization_func, test_images: List[np.ndarray]):
    """Profile the performance of ML pipeline components"""
    
    profiler = PerformanceProfiler()
    
    for i, image in enumerate(test_images):
        # Time Mask R-CNN inference
        profiler.time_operation("mask_rcnn_inference", mask_rcnn_func, image)
        
        # Simulate regularization timing
        dummy_polygons = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
        profiler.time_operation("geometric_regularization", regularization_func, dummy_polygons)
    
    return profiler.get_performance_summary()