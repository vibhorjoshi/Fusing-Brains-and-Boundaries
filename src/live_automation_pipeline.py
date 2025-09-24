"""
Live Automated End-to-End Building Footprint Extraction Pipeline

This module provides a complete automation demo showing:
1. Image → Patches (configurable grid)
2. Initial Masking on each patch
3. Mask R-CNN processing on large scale
4. Post-processing and segmentation
5. RR, FER, RT regularization
6. Adaptive Fusion decision making
7. IoU calculation and iterative improvement

Usage:
    pipeline = LiveAutomationPipeline()
    pipeline.run_live_demo()
"""

import numpy as np
import cv2
import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import threading
from queue import Queue
import json


@dataclass
class PipelineStage:
    """Represents a stage in the processing pipeline"""
    name: str
    input_data: Any
    output_data: Any
    processing_time: float
    metrics: Dict[str, float]
    status: str = "pending"  # pending, processing, completed


@dataclass 
class PatchInfo:
    """Information about an image patch"""
    patch_id: int
    x: int
    y: int
    width: int
    height: int
    patch_data: np.ndarray
    has_buildings: bool = False
    confidence: float = 0.0
    iou_score: float = 0.0


class LiveAutomationPipeline:
    """Live automation pipeline for building footprint extraction"""
    
    def __init__(self, patch_grid_size: int = 3):
        """Initialize the automation pipeline
        
        Args:
            patch_grid_size: Grid size for patch division (3 = 3x3 = 9 patches)
        """
        self.patch_grid_size = patch_grid_size
        self.total_patches = patch_grid_size * patch_grid_size
        
        # Pipeline stages
        self.stages = [
            "🔷 Input Image Loading",
            "📐 Patch Division", 
            "🎯 Initial Masking",
            "🤖 Mask R-CNN Processing",
            "⚙️ Post-Processing",
            "🔧 RR Regularization",
            "🛠️ FER Regularization", 
            "⭕ RT Regularization",
            "🧠 Adaptive Fusion",
            "📊 IoU Calculation",
            "🔄 Iterative Improvement"
        ]
        
        # Results storage
        self.current_image = None
        self.ground_truth = None
        self.patches: List[PatchInfo] = []
        self.stage_results = {}
        self.fusion_history = []
        self.iou_history = []
        
        # Processing parameters
        self.iteration_count = 0
        self.max_iterations = 5
        self.target_iou = 0.85
        
        # Synthetic data for demo
        self.demo_images = self._create_demo_images()
        
    def _create_demo_images(self) -> List[Dict[str, np.ndarray]]:
        """Create synthetic demo images with ground truth"""
        demo_images = []
        
        for i in range(5):
            # Create synthetic satellite image
            img = self._create_synthetic_satellite_image(640, 640)
            
            # Create corresponding ground truth
            gt = self._create_ground_truth_mask(640, 640)
            
            demo_images.append({
                'image': img,
                'ground_truth': gt,
                'name': f'Demo_City_{i+1}'
            })
            
        return demo_images
    
    def _create_synthetic_satellite_image(self, width: int, height: int) -> np.ndarray:
        """Create realistic synthetic satellite image"""
        # Base terrain
        image = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
        
        # Add roads (grid pattern)
        road_color = (60, 60, 60)
        for x in range(0, width, 80):
            cv2.line(image, (x, 0), (x, height), road_color, 3)
        for y in range(0, height, 80):
            cv2.line(image, (0, y), (width, y), road_color, 3)
        
        # Add buildings
        building_positions = []
        for _ in range(random.randint(15, 25)):
            bw = random.randint(20, 50)
            bh = random.randint(20, 50)
            x = random.randint(0, width - bw)
            y = random.randint(0, height - bh)
            
            # Building color
            color = (random.randint(40, 80), random.randint(40, 80), random.randint(40, 80))
            cv2.rectangle(image, (x, y), (x + bw, y + bh), color, -1)
            
            # Add shadow
            shadow_color = tuple(max(0, c - 20) for c in color)
            cv2.rectangle(image, (x + 2, y + 2), (x + bw + 2, y + bh + 2), shadow_color, 2)
            
            building_positions.append((x, y, bw, bh))
        
        # Add some vegetation (green areas)
        for _ in range(random.randint(5, 10)):
            x, y = random.randint(0, width-40), random.randint(0, height-40)
            size = random.randint(20, 40)
            color = (random.randint(60, 100), random.randint(100, 140), random.randint(60, 100))
            cv2.circle(image, (x, y), size, color, -1)
        
        # Store building positions for ground truth
        self._building_positions = building_positions
        
        return image
    
    def _create_ground_truth_mask(self, width: int, height: int) -> np.ndarray:
        """Create ground truth mask for synthetic image"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Use building positions from synthetic image creation
        if hasattr(self, '_building_positions'):
            for x, y, bw, bh in self._building_positions:
                cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        
        return mask
    
    def run_live_demo(self) -> Dict[str, Any]:
        """Run the complete live automation demo"""
        # Select demo image
        demo_data = random.choice(self.demo_images)
        self.current_image = demo_data['image']
        self.ground_truth = demo_data['ground_truth']
        
        print(f"🚀 Starting Live Automation Demo: {demo_data['name']}")
        print(f"📊 Pipeline: {len(self.stages)} stages, {self.total_patches} patches")
        
        # Initialize results
        results = {
            'demo_name': demo_data['name'],
            'pipeline_stages': [],
            'final_metrics': {},
            'processing_timeline': []
        }
        
        start_time = time.time()
        
        # Stage 1: Input Image Loading
        stage_result = self._stage_input_loading()
        results['pipeline_stages'].append(stage_result)
        
        # Stage 2: Patch Division
        stage_result = self._stage_patch_division()
        results['pipeline_stages'].append(stage_result)
        
        # Stage 3: Initial Masking
        stage_result = self._stage_initial_masking()
        results['pipeline_stages'].append(stage_result)
        
        # Stage 4: Mask R-CNN Processing
        stage_result = self._stage_mask_rcnn()
        results['pipeline_stages'].append(stage_result)
        
        # Stage 5: Post-Processing
        stage_result = self._stage_post_processing()
        results['pipeline_stages'].append(stage_result)
        
        # Stage 6-8: Regularization (RR, FER, RT)
        rr_result = self._stage_rr_regularization()
        fer_result = self._stage_fer_regularization() 
        rt_result = self._stage_rt_regularization()
        
        results['pipeline_stages'].extend([rr_result, fer_result, rt_result])
        
        # Stage 9: Adaptive Fusion (Iterative)
        fusion_result = self._stage_adaptive_fusion_iterative()
        results['pipeline_stages'].append(fusion_result)
        
        # Stage 10: Final IoU Calculation
        iou_result = self._stage_final_iou_calculation()
        results['pipeline_stages'].append(iou_result)
        
        # Calculate final metrics
        total_time = time.time() - start_time
        results['final_metrics'] = {
            'total_processing_time': total_time,
            'final_iou': self.iou_history[-1] if self.iou_history else 0.0,
            'iterations_completed': self.iteration_count,
            'patches_processed': len(self.patches),
            'improvement_rate': self._calculate_improvement_rate()
        }
        
        print(f"✅ Pipeline Complete! Final IoU: {results['final_metrics']['final_iou']:.3f}")
        
        return results
    
    def _stage_input_loading(self) -> PipelineStage:
        """Stage 1: Input Image Loading"""
        print("🔷 Stage 1: Loading input image...")
        time.sleep(0.5)  # Simulate processing time
        
        metrics = {
            'image_size': f"{self.current_image.shape[1]}x{self.current_image.shape[0]}",
            'channels': self.current_image.shape[2],
            'memory_usage_mb': self.current_image.nbytes / (1024*1024)
        }
        
        return PipelineStage(
            name="Input Loading",
            input_data="Raw satellite image",
            output_data="Loaded image tensor",
            processing_time=0.5,
            metrics=metrics,
            status="completed"
        )
    
    def _stage_patch_division(self) -> PipelineStage:
        """Stage 2: Divide image into patches"""
        print(f"📐 Stage 2: Dividing into {self.patch_grid_size}x{self.patch_grid_size} patches...")
        time.sleep(0.3)
        
        h, w = self.current_image.shape[:2]
        patch_h, patch_w = h // self.patch_grid_size, w // self.patch_grid_size
        
        self.patches = []
        patch_id = 0
        
        for i in range(self.patch_grid_size):
            for j in range(self.patch_grid_size):
                x = j * patch_w
                y = i * patch_h
                
                # Extract patch
                patch_data = self.current_image[y:y+patch_h, x:x+patch_w]
                
                # Create patch info
                patch_info = PatchInfo(
                    patch_id=patch_id,
                    x=x, y=y,
                    width=patch_w,
                    height=patch_h,
                    patch_data=patch_data
                )
                
                self.patches.append(patch_info)
                patch_id += 1
        
        metrics = {
            'total_patches': len(self.patches),
            'patch_size': f"{patch_w}x{patch_h}",
            'coverage': 'Complete'
        }
        
        return PipelineStage(
            name="Patch Division",
            input_data="Full image",
            output_data=f"{len(self.patches)} patches",
            processing_time=0.3,
            metrics=metrics,
            status="completed"
        )
    
    def _stage_initial_masking(self) -> PipelineStage:
        """Stage 3: Initial masking on patches"""
        print("🎯 Stage 3: Applying initial masking to each patch...")
        
        processed_patches = 0
        total_building_pixels = 0
        
        for patch in self.patches:
            time.sleep(0.1)  # Simulate processing
            
            # Simple thresholding for initial mask
            gray = cv2.cvtColor(patch.patch_data, cv2.COLOR_BGR2GRAY)
            
            # Buildings are typically darker than background
            mean_intensity = np.mean(gray)
            threshold = max(50, mean_intensity - 20)
            
            _, initial_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Store initial mask
            patch.initial_mask = initial_mask
            
            # Check for building presence
            building_pixels = np.sum(initial_mask > 0)
            total_building_pixels += building_pixels
            
            # Heuristic for building presence
            patch.has_buildings = building_pixels > (patch.width * patch.height * 0.05)
            patch.confidence = min(1.0, building_pixels / (patch.width * patch.height * 0.3))
            
            processed_patches += 1
            print(f"  📦 Patch {patch.patch_id + 1}/{len(self.patches)}: "
                  f"Buildings: {'✓' if patch.has_buildings else '✗'} "
                  f"(Confidence: {patch.confidence:.2f})")
        
        metrics = {
            'patches_processed': processed_patches,
            'patches_with_buildings': sum(1 for p in self.patches if p.has_buildings),
            'total_building_pixels': int(total_building_pixels),
            'average_confidence': np.mean([p.confidence for p in self.patches])
        }
        
        return PipelineStage(
            name="Initial Masking",
            input_data="Image patches",
            output_data="Initial masks",
            processing_time=processed_patches * 0.1,
            metrics=metrics,
            status="completed"
        )
    
    def _stage_mask_rcnn(self) -> PipelineStage:
        """Stage 4: Mask R-CNN processing on large scale"""
        print("🤖 Stage 4: Running Mask R-CNN on full image...")
        time.sleep(1.5)  # Simulate neural network processing
        
        # Simulate Mask R-CNN detection
        h, w = self.current_image.shape[:2]
        mask_rcnn_result = np.zeros((h, w), dtype=np.uint8)
        
        # Generate realistic building detections
        detected_buildings = 0
        
        for patch in self.patches:
            if patch.has_buildings:
                # Simulate R-CNN detection in patch area
                mask_region = self._simulate_rcnn_detection(patch)
                
                # Place in full image
                y_end = min(patch.y + patch.height, h)
                x_end = min(patch.x + patch.width, w)
                
                mask_rcnn_result[patch.y:y_end, patch.x:x_end] = np.maximum(
                    mask_rcnn_result[patch.y:y_end, patch.x:x_end],
                    mask_region[:y_end-patch.y, :x_end-patch.x]
                )
                
                detected_buildings += 1
        
        self.stage_results['mask_rcnn'] = mask_rcnn_result
        
        metrics = {
            'buildings_detected': detected_buildings,
            'detection_coverage': f"{np.sum(mask_rcnn_result > 0)} pixels",
            'processing_method': 'Simulated Mask R-CNN',
            'inference_time': '1.5s'
        }
        
        return PipelineStage(
            name="Mask R-CNN",
            input_data="Full image + patch hints",
            output_data="Building detection masks",
            processing_time=1.5,
            metrics=metrics,
            status="completed"
        )
    
    def _simulate_rcnn_detection(self, patch: PatchInfo) -> np.ndarray:
        """Simulate R-CNN detection for a patch"""
        mask = np.zeros((patch.height, patch.width), dtype=np.uint8)
        
        # Use initial mask as base but add some refinement
        if hasattr(patch, 'initial_mask'):
            # Apply morphological operations to simulate R-CNN refinement
            kernel = np.ones((3, 3), np.uint8)
            refined = cv2.morphologyEx(patch.initial_mask, cv2.MORPH_CLOSE, kernel)
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
            
            # Add some random variations to simulate neural network
            noise = np.random.randint(-20, 20, refined.shape).astype(np.int16)
            refined = np.clip(refined.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            mask = refined
        
        return mask
    
    def _stage_post_processing(self) -> PipelineStage:
        """Stage 5: Post-processing and segmentation refinement"""
        print("⚙️ Stage 5: Post-processing segmentation...")
        time.sleep(0.8)
        
        mask_rcnn = self.stage_results['mask_rcnn']
        
        # Apply post-processing operations
        kernel = np.ones((5, 5), np.uint8)
        
        # Remove noise
        denoised = cv2.medianBlur(mask_rcnn, 5)
        
        # Fill holes
        filled = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Remove small objects
        cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
        
        self.stage_results['post_processed'] = cleaned
        
        # Calculate improvement metrics
        original_objects = self._count_objects(mask_rcnn)
        cleaned_objects = self._count_objects(cleaned)
        
        metrics = {
            'objects_before': original_objects,
            'objects_after': cleaned_objects,
            'noise_removed': max(0, original_objects - cleaned_objects),
            'processing_operations': 'Median Filter + Morphology'
        }
        
        return PipelineStage(
            name="Post-Processing",
            input_data="Raw R-CNN masks",
            output_data="Clean segmentation",
            processing_time=0.8,
            metrics=metrics,
            status="completed"
        )
    
    def _count_objects(self, mask: np.ndarray) -> int:
        """Count number of objects in binary mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)
    
    def _stage_rr_regularization(self) -> PipelineStage:
        """Stage 6: RR (Ridge Regression) Regularization"""
        print("🔧 Stage 6: Applying RR regularization...")
        time.sleep(0.6)
        
        base_mask = self.stage_results['post_processed']
        
        # Simulate RR regularization (smoothing with L2 penalty)
        rr_result = self._apply_gaussian_regularization(base_mask, sigma=1.0)
        
        self.stage_results['rr_regularized'] = rr_result
        
        iou_vs_gt = self._calculate_iou(rr_result, self.ground_truth)
        
        metrics = {
            'regularization_type': 'Ridge Regression (L2)',
            'smoothing_sigma': 1.0,
            'iou_vs_ground_truth': iou_vs_gt,
            'edge_preservation': 'Medium'
        }
        
        return PipelineStage(
            name="RR Regularization",
            input_data="Post-processed mask",
            output_data="RR regularized mask",
            processing_time=0.6,
            metrics=metrics,
            status="completed"
        )
    
    def _stage_fer_regularization(self) -> PipelineStage:
        """Stage 7: FER (Feature Enhancement Regularization)"""
        print("🛠️ Stage 7: Applying FER regularization...")
        time.sleep(0.7)
        
        base_mask = self.stage_results['post_processed']
        
        # Simulate FER (edge-preserving smoothing)
        fer_result = self._apply_bilateral_regularization(base_mask)
        
        self.stage_results['fer_regularized'] = fer_result
        
        iou_vs_gt = self._calculate_iou(fer_result, self.ground_truth)
        
        metrics = {
            'regularization_type': 'Feature Enhancement',
            'edge_preservation': 'High',
            'iou_vs_ground_truth': iou_vs_gt,
            'bilateral_sigma': '(50, 50)'
        }
        
        return PipelineStage(
            name="FER Regularization", 
            input_data="Post-processed mask",
            output_data="FER regularized mask",
            processing_time=0.7,
            metrics=metrics,
            status="completed"
        )
    
    def _stage_rt_regularization(self) -> PipelineStage:
        """Stage 8: RT (Robust Thresholding) Regularization"""
        print("⭕ Stage 8: Applying RT regularization...")
        time.sleep(0.5)
        
        base_mask = self.stage_results['post_processed']
        
        # Simulate RT (adaptive thresholding)
        rt_result = self._apply_adaptive_regularization(base_mask)
        
        self.stage_results['rt_regularized'] = rt_result
        
        iou_vs_gt = self._calculate_iou(rt_result, self.ground_truth)
        
        metrics = {
            'regularization_type': 'Robust Thresholding',
            'adaptation_method': 'Local statistics',
            'iou_vs_ground_truth': iou_vs_gt,
            'threshold_adaptation': 'Dynamic'
        }
        
        return PipelineStage(
            name="RT Regularization",
            input_data="Post-processed mask", 
            output_data="RT regularized mask",
            processing_time=0.5,
            metrics=metrics,
            status="completed"
        )
    
    def _apply_gaussian_regularization(self, mask: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian smoothing regularization"""
        smoothed = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
        return (smoothed > 127).astype(np.uint8) * 255
    
    def _apply_bilateral_regularization(self, mask: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering regularization"""
        filtered = cv2.bilateralFilter(mask, 9, 50, 50)
        return filtered
    
    def _apply_adaptive_regularization(self, mask: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding regularization"""
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            gray = mask
        
        # Apply adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return adaptive
    
    def _stage_adaptive_fusion_iterative(self) -> PipelineStage:
        """Stage 9: Adaptive Fusion with iterative improvement"""
        print("🧠 Stage 9: Running adaptive fusion with iterations...")
        
        # Get all regularization results
        candidates = {
            'rr': self.stage_results['rr_regularized'],
            'fer': self.stage_results['fer_regularized'], 
            'rt': self.stage_results['rt_regularized']
        }
        
        best_result = None
        best_iou = 0.0
        
        # Iterative improvement
        for iteration in range(self.max_iterations):
            print(f"  🔄 Iteration {iteration + 1}/{self.max_iterations}")
            time.sleep(0.4)
            
            # Adaptive fusion decision
            fusion_result, fusion_iou = self._adaptive_fusion_step(candidates)
            
            self.iou_history.append(fusion_iou)
            
            print(f"    📊 IoU: {fusion_iou:.3f}")
            
            if fusion_iou > best_iou:
                best_result = fusion_result.copy()
                best_iou = fusion_iou
            
            # Update candidates based on best result
            if fusion_iou > self.target_iou:
                print(f"    ✅ Target IoU {self.target_iou} reached!")
                break
            
            # Refine candidates for next iteration
            candidates = self._refine_candidates(candidates, best_result)
            
            self.iteration_count = iteration + 1
        
        self.stage_results['final_fusion'] = best_result
        
        metrics = {
            'iterations_completed': self.iteration_count,
            'best_iou_achieved': best_iou,
            'target_iou': self.target_iou,
            'convergence_rate': self._calculate_convergence_rate(),
            'final_method': self._determine_best_method(candidates)
        }
        
        return PipelineStage(
            name="Adaptive Fusion",
            input_data="RR + FER + RT results",
            output_data="Optimized fusion result",
            processing_time=self.iteration_count * 0.4,
            metrics=metrics,
            status="completed"
        )
    
    def _adaptive_fusion_step(self, candidates: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """Perform one step of adaptive fusion"""
        
        # Calculate IoU for each candidate
        ious = {}
        for name, mask in candidates.items():
            iou = self._calculate_iou(mask, self.ground_truth)
            ious[name] = iou
        
        # Weighted fusion based on IoU performance
        total_iou = sum(ious.values())
        
        if total_iou == 0:
            # Equal weights if all IoUs are 0
            weights = {name: 1.0/len(candidates) for name in candidates.keys()}
        else:
            # IoU-based weights
            weights = {name: iou/total_iou for name, iou in ious.items()}
        
        # Fusion
        fused_result = np.zeros_like(list(candidates.values())[0], dtype=np.float32)
        
        for name, mask in candidates.items():
            weight = weights[name]
            fused_result += weight * mask.astype(np.float32)
        
        # Convert back to binary
        fused_binary = (fused_result > 127).astype(np.uint8) * 255
        
        # Calculate final IoU
        final_iou = self._calculate_iou(fused_binary, self.ground_truth)
        
        return fused_binary, final_iou
    
    def _refine_candidates(self, candidates: Dict[str, np.ndarray], best_result: np.ndarray) -> Dict[str, np.ndarray]:
        """Refine candidates based on best result so far"""
        refined = {}
        
        for name, mask in candidates.items():
            # Blend with best result to improve
            alpha = 0.7  # Weight for current mask
            beta = 0.3   # Weight for best result
            
            blended = cv2.addWeighted(mask, alpha, best_result, beta, 0)
            refined[name] = blended
        
        return refined
    
    def _calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU)"""
        # Ensure binary masks
        pred_binary = (pred_mask > 127).astype(np.uint8)
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
        union = np.sum((pred_binary == 1) | (gt_binary == 1))
        
        if union == 0:
            return 1.0  # Perfect match if both are empty
        
        iou = intersection / union
        return float(iou)
    
    def _stage_final_iou_calculation(self) -> PipelineStage:
        """Stage 10: Final IoU calculation and metrics"""
        print("📊 Stage 10: Calculating final metrics...")
        time.sleep(0.3)
        
        final_result = self.stage_results['final_fusion']
        final_iou = self._calculate_iou(final_result, self.ground_truth)
        
        # Additional metrics
        precision = self._calculate_precision(final_result, self.ground_truth)
        recall = self._calculate_recall(final_result, self.ground_truth)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'final_iou': final_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'pixel_accuracy': self._calculate_pixel_accuracy(final_result, self.ground_truth),
            'improvement_from_initial': final_iou - (self.iou_history[0] if self.iou_history else 0)
        }
        
        print(f"    📈 Final Results:")
        print(f"       IoU: {final_iou:.3f}")
        print(f"       Precision: {precision:.3f}")  
        print(f"       Recall: {recall:.3f}")
        print(f"       F1-Score: {f1_score:.3f}")
        
        return PipelineStage(
            name="Final Metrics",
            input_data="Fused result + Ground truth",
            output_data="Performance metrics",
            processing_time=0.3,
            metrics=metrics,
            status="completed"
        )
    
    def _calculate_precision(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate precision"""
        pred_binary = (pred_mask > 127).astype(np.uint8)
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        true_positive = np.sum((pred_binary == 1) & (gt_binary == 1))
        predicted_positive = np.sum(pred_binary == 1)
        
        if predicted_positive == 0:
            return 0.0
        
        return float(true_positive / predicted_positive)
    
    def _calculate_recall(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate recall"""
        pred_binary = (pred_mask > 127).astype(np.uint8)
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        true_positive = np.sum((pred_binary == 1) & (gt_binary == 1))
        actual_positive = np.sum(gt_binary == 1)
        
        if actual_positive == 0:
            return 0.0
        
        return float(true_positive / actual_positive)
    
    def _calculate_pixel_accuracy(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate pixel-wise accuracy"""
        pred_binary = (pred_mask > 127).astype(np.uint8)
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        correct_pixels = np.sum(pred_binary == gt_binary)
        total_pixels = pred_binary.size
        
        return float(correct_pixels / total_pixels)
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate over iterations"""
        if len(self.iou_history) < 2:
            return 0.0
        
        initial_iou = self.iou_history[0]
        final_iou = self.iou_history[-1]
        
        if initial_iou == 0:
            return float(final_iou)
        
        return float((final_iou - initial_iou) / initial_iou)
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate"""
        if len(self.iou_history) < 2:
            return 0.0
        
        # Calculate how quickly IoU improved
        improvements = []
        for i in range(1, len(self.iou_history)):
            improvement = self.iou_history[i] - self.iou_history[i-1]
            improvements.append(improvement)
        
        return float(np.mean(improvements)) if improvements else 0.0
    
    def _determine_best_method(self, candidates: Dict[str, np.ndarray]) -> str:
        """Determine which regularization method performed best"""
        best_method = "fusion"
        best_iou = 0.0
        
        for name, mask in candidates.items():
            iou = self._calculate_iou(mask, self.ground_truth)
            if iou > best_iou:
                best_iou = iou
                best_method = name
        
        return best_method
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization"""
        return {
            'original_image': self.current_image,
            'ground_truth': self.ground_truth,
            'patches': self.patches,
            'stage_results': self.stage_results,
            'iou_history': self.iou_history,
            'fusion_history': self.fusion_history
        }


# Demo runner function
def run_automation_demo(patch_size: int = 3) -> Dict[str, Any]:
    """Run the live automation demo
    
    Args:
        patch_size: Patch grid size (3 = 3x3 patches)
    
    Returns:
        Complete demo results
    """
    pipeline = LiveAutomationPipeline(patch_grid_size=patch_size)
    results = pipeline.run_live_demo()
    
    # Add visualization data
    results['visualization_data'] = pipeline.get_visualization_data()
    
    return results


if __name__ == "__main__":
    # Run demo
    print("🚀 Starting Live Building Footprint Extraction Demo")
    print("=" * 60)
    
    demo_results = run_automation_demo(patch_size=3)
    
    print("\n" + "=" * 60)
    print("📊 DEMO COMPLETE - Summary:")
    print(f"   Final IoU: {demo_results['final_metrics']['final_iou']:.3f}")
    print(f"   Processing Time: {demo_results['final_metrics']['total_processing_time']:.1f}s")
    print(f"   Iterations: {demo_results['final_metrics']['iterations_completed']}")
    print(f"   Improvement Rate: {demo_results['final_metrics']['improvement_rate']:.1%}")