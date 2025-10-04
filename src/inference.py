import torch
import numpy as np
from PIL import Image
import time
import traceback
from typing import Dict, Any, Optional

class BuildingFootprintExtractor:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        self.model = None
        self.is_initialized = False
        print("Building footprint extractor initialized in demo mode")
    
    def extract(self, image, method="RL Fusion") -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            if isinstance(image, np.ndarray):
                input_array = image
            else:
                input_array = np.array(image)
            
            height, width = input_array.shape[:2]
            channels = input_array.shape[2] if len(input_array.shape) > 2 else 1
            
            # Demo mode visualization
            viz_image = self._create_demo_visualization(input_array, method)
            processing_time = time.time() - start_time
            
            return {
                "visualization": viz_image,
                "status": "Demo Mode",
                "input_shape": f"{height}x{width}x{channels}",
                "buildings_count": 5,
                "processing_time": f"{processing_time:.2f}s",
                "method_used": method,
                "additional_info": f"Demo simulation of {method} method"
            }
            
        except Exception as e:
            return {
                "visualization": input_array,
                "status": "Error",
                "input_shape": "Unknown",
                "buildings_count": 0,
                "processing_time": "0.00s",
                "method_used": method,
                "additional_info": f"Error: {str(e)}"
            }
    
    def _create_demo_visualization(self, image, method):
        try:
            import cv2
            viz_image = image.copy()
            height, width = viz_image.shape[:2]
            
            # Add random rectangles
            for i in range(3):
                x1 = np.random.randint(0, max(1, width - 100))
                y1 = np.random.randint(0, max(1, height - 100))
                x2 = x1 + np.random.randint(30, 80)
                y2 = y1 + np.random.randint(30, 80)
                
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(viz_image, f"{method} (Demo)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return viz_image
        except:
            return image
