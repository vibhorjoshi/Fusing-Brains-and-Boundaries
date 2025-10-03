#!/usr/bin/env python3
"""
Complete Pipeline Workflow - Real USA Agricultural Detection System
This script integrates all components of the GeoAI pipeline:
- Preprocessing of satellite images
- MaskRCNN for 3D mask creation
- Post-processing and adaptive fusion
- Redis storage for live results
- GPU performance monitoring
- Integration with GeoAI library for enhanced analysis
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import tensorflow as tf
import fakeredis
import psutil
import threading
from datetime import datetime

# Import the GeoAI library
try:
    from src.open_source_geo_ai import OpenSourceGeoAI
    GEOAI_AVAILABLE = True
    print("GeoAI library loaded successfully")
except ImportError as e:
    GEOAI_AVAILABLE = False
    print(f"GeoAI library not available: {e}")
    print("Some advanced functions will be limited")

# Configure TensorFlow to use GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU for processing.")

# Initialize Redis client (using fakeredis for development)
redis_client = fakeredis.FakeStrictRedis()
print("Redis client initialized")

# Simulated pipeline components
class PreprocessingModule:
    """Handles preprocessing of satellite imagery"""
    
    def __init__(self):
        print("Preprocessing module initialized")
        
    def process(self, image_path):
        """Process a satellite image"""
        try:
            # Load image
            if isinstance(image_path, str) and os.path.exists(image_path):
                image = cv2.imread(image_path)
            elif isinstance(image_path, np.ndarray):
                image = image_path
            else:
                # Generate random test image if path doesn't exist
                image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            # Apply preprocessing steps
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Simulated normalization
            image = image / 255.0
            
            print(f"Image preprocessed: {image.shape}")
            return image
        
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None


class MaskRCNNModule:
    """Handles 3D mask creation using MaskRCNN"""
    
    def __init__(self):
        print("MaskRCNN module initialized")
        # Simulated model loading
        time.sleep(1)
        
    def create_masks(self, image):
        """Create 3D masks for agricultural detection"""
        try:
            if image is None:
                return None
                
            # Simulated mask generation
            height, width = image.shape[:2]
            
            # Generate random binary masks (would be MaskRCNN output in production)
            num_masks = np.random.randint(3, 8)
            masks = []
            
            for i in range(num_masks):
                # Create a random polygon-like mask
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Generate random polygon points
                num_points = np.random.randint(6, 15)
                points = np.random.randint(0, min(height, width), (num_points, 2))
                
                # Draw filled polygon
                cv2.fillPoly(mask, [points], 1)
                
                masks.append(mask)
            
            # Stack masks into 3D array
            masks_3d = np.stack(masks, axis=2)
            print(f"Created {num_masks} masks with shape {masks_3d.shape}")
            
            return masks_3d
            
        except Exception as e:
            print(f"Error in MaskRCNN processing: {e}")
            return None


class AdaptiveFusionModule:
    """Handles adaptive fusion of masks and original image"""
    
    def __init__(self):
        print("Adaptive Fusion module initialized")
    
    def process(self, image, masks):
        """Apply adaptive fusion algorithm"""
        try:
            if image is None or masks is None:
                return None
                
            # Ensure image is float and in range [0, 1]
            if image.dtype != np.float32 and image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
                
            # Create visualization by blending masks with original image
            height, width = image.shape[:2]
            num_masks = masks.shape[2]
            
            # Create random colors for visualization
            colors = np.random.randint(0, 255, (num_masks, 3))
            
            # Create output visualization
            output = image.copy()
            
            # Apply each mask with a different color
            for i in range(num_masks):
                mask = masks[:, :, i]
                color = colors[i] / 255.0
                
                for c in range(3):
                    output[:, :, c] = np.where(mask == 1, 
                                             0.7 * output[:, :, c] + 0.3 * color[c], 
                                             output[:, :, c])
            
            print("Adaptive fusion applied successfully")
            return output
            
        except Exception as e:
            print(f"Error in adaptive fusion: {e}")
            return None


class PostProcessingModule:
    """Handles post-processing of results"""
    
    def __init__(self):
        print("Post-processing module initialized")
    
    def process(self, image, masks):
        """Apply post-processing to results"""
        try:
            if image is None or masks is None:
                return None, None
                
            # Extract agricultural regions
            height, width = image.shape[:2]
            num_masks = masks.shape[2]
            
            # Generate simulated agricultural data
            agricultural_data = []
            
            for i in range(num_masks):
                mask = masks[:, :, i]
                area = np.sum(mask)
                
                # Calculate bounding rectangle
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Generate random crop type and health score
                crop_types = ["Corn", "Soybeans", "Wheat", "Cotton", "Rice"]
                crop_type = np.random.choice(crop_types)
                health_score = np.random.uniform(0.5, 1.0)
                
                agricultural_data.append({
                    "id": i,
                    "crop_type": crop_type,
                    "area_px": int(area),
                    "area_acres": int(area / 100),  # Simulated conversion
                    "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "health_score": float(health_score),
                    "timestamp": datetime.now().isoformat()
                })
            
            print(f"Post-processing complete: {len(agricultural_data)} regions identified")
            return image, agricultural_data
            
        except Exception as e:
            print(f"Error in post-processing: {e}")
            return None, None


class PerformanceMonitor:
    """Monitors system performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "timestamps": []
        }
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            # Collect CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # Record metrics
            self.metrics["cpu_usage"].append(cpu_percent)
            self.metrics["memory_usage"].append(memory_percent)
            self.metrics["timestamps"].append(datetime.now().isoformat())
            
            # Log metrics every 10 recordings
            if len(self.metrics["cpu_usage"]) % 10 == 0:
                avg_cpu = sum(self.metrics["cpu_usage"][-10:]) / 10
                avg_mem = sum(self.metrics["memory_usage"][-10:]) / 10
                print(f"Performance: CPU {avg_cpu:.1f}%, Memory {avg_mem:.1f}%")
            
            # Sleep before next recording
            time.sleep(1.0)
    
    def get_metrics(self):
        """Get current performance metrics"""
        return self.metrics


class PipelineManager:
    """Manages the complete pipeline workflow"""
    
    def __init__(self):
        # Initialize pipeline components
        self.preprocessing = PreprocessingModule()
        self.maskrcnn = MaskRCNNModule()
        self.adaptive_fusion = AdaptiveFusionModule()
        self.postprocessing = PostProcessingModule()
        self.monitor = PerformanceMonitor()
        
        # Initialize GeoAI client if available
        self.geoai = None
        if GEOAI_AVAILABLE:
            try:
                self.geoai = OpenSourceGeoAI()
                print("GeoAI client initialized for enhanced analysis")
            except Exception as e:
                print(f"Error initializing GeoAI client: {e}")
        
        # Start performance monitoring
        self.monitor.start_monitoring()
        print("Pipeline manager initialized")
        
    def process_image(self, image_path):
        """Process a single image through the complete pipeline"""
        try:
            print(f"\nProcessing image: {image_path}")
            
            # Step 1: Preprocessing
            start_time = time.time()
            preprocessed = self.preprocessing.process(image_path)
            preprocess_time = time.time() - start_time
            
            # Step 2: MaskRCNN for 3D mask creation
            start_time = time.time()
            masks = self.maskrcnn.create_masks(preprocessed)
            maskrcnn_time = time.time() - start_time
            
            # Step 3: Adaptive Fusion
            start_time = time.time()
            fused = self.adaptive_fusion.process(preprocessed, masks)
            fusion_time = time.time() - start_time
            
            # Step 4: Post-processing
            start_time = time.time()
            result_image, agricultural_data = self.postprocessing.process(fused, masks)
            postprocess_time = time.time() - start_time
            
            # Calculate total processing time
            total_time = preprocess_time + maskrcnn_time + fusion_time + postprocess_time
            
            # Store results in Redis
            self._store_results_in_redis(image_path, agricultural_data, 
                                        {
                                            "preprocess_time": preprocess_time,
                                            "maskrcnn_time": maskrcnn_time,
                                            "fusion_time": fusion_time,
                                            "postprocess_time": postprocess_time,
                                            "total_time": total_time
                                        })
            
            print(f"Pipeline complete in {total_time:.2f}s")
            return result_image, agricultural_data
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            return None, None
    
    def _store_results_in_redis(self, image_path, agricultural_data, timing_data):
        """Store processing results in Redis"""
        try:
            # Create a unique key based on image path and timestamp
            key_base = f"agri:{os.path.basename(image_path).split('.')[0]}:{int(time.time())}"
            
            # Store agricultural data
            if agricultural_data:
                redis_client.set(f"{key_base}:data", json.dumps(agricultural_data))
            
            # Store timing data
            redis_client.set(f"{key_base}:timing", json.dumps(timing_data))
            
            # Store performance metrics
            metrics = self.monitor.get_metrics()
            if metrics["cpu_usage"]:
                latest_metrics = {
                    "cpu": metrics["cpu_usage"][-1],
                    "memory": metrics["memory_usage"][-1],
                    "timestamp": metrics["timestamps"][-1]
                }
                redis_client.set(f"{key_base}:metrics", json.dumps(latest_metrics))
            
            print(f"Results stored in Redis with key prefix: {key_base}")
            
        except Exception as e:
            print(f"Error storing results in Redis: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.monitor.stop_monitoring()
        print("Pipeline resources cleaned up")


def run_demo(num_images=5):
    """Run a demo with simulated images"""
    pipeline = PipelineManager()
    
    try:
        for i in range(num_images):
            # Generate a random image path (or use real path if available)
            image_path = f"sample_image_{i+1}.jpg"
            
            # Process the image
            result_image, agricultural_data = pipeline.process_image(image_path)
            
            # Print summary of detected agricultural regions
            if agricultural_data:
                print(f"\nDetected {len(agricultural_data)} agricultural regions:")
                for j, region in enumerate(agricultural_data[:3]):  # Show first 3 only
                    print(f"  Region {j+1}: {region['crop_type']}, "
                          f"Area: {region['area_acres']} acres, "
                          f"Health: {region['health_score']:.2f}")
                if len(agricultural_data) > 3:
                    print(f"  ... and {len(agricultural_data) - 3} more regions")
            
            # Simulate delay between images
            time.sleep(2)
    
    finally:
        # Clean up resources
        pipeline.cleanup()


if __name__ == "__main__":
    print("Starting Real USA Agricultural Detection System")
    print("=" * 80)
    run_demo()
    print("=" * 80)
    print("Demo complete")