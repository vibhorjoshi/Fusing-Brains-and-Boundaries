"""
GPU-Accelerated Regularizers for Building Footprint Processing

This module implements three regularization techniques optimized for GPU execution:
1. RT (Regular Topology) - Uses morphological closing to straighten boundaries
2. RR (Regular Rectangle) - Uses opening then closing to remove noise and maintain shape
3. FER (Feature Edge Regularization) - Edge-aware dilation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
import cv2
from kornia.morphology import dilation, erosion, closing, opening
from kornia.filters import canny


class GPURegularizer:
    """GPU-accelerated implementation of the HybridRegularizer.
    
    This class provides GPU-accelerated versions of RT, RR, and FER regularization
    techniques for building footprint masks.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create GPU kernels for morphological operations
        self.kernel_rt = torch.ones(3, 3, device=self.device)
        self.kernel_rr = torch.ones(5, 5, device=self.device)
        self.kernel_edge = torch.ones(3, 3, device=self.device)
        
    def apply(self, mask_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply all regularization techniques to a batch of masks.
        
        Args:
            mask_batch: [B,1,H,W] torch tensor on GPU with values 0-1
            
        Returns:
            Dictionary with regularized variants ('original', 'rt', 'rr', 'fer')
        """
        # Ensure binary float mask
        m = (mask_batch > 0.5).float()
        batch_size = m.shape[0]
        results = {"original": m}
        
        # Apply all regularization types using GPU operations
        results["rt"] = self._apply_rt(m)
        results["rr"] = self._apply_rr(m)
        results["fer"] = self._apply_fer(m)
        
        return results
    
    def _apply_rt(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply RT regularization (mild closing to straighten boundaries).
        
        Args:
            mask: [B,1,H,W] torch tensor on GPU
            
        Returns:
            RT-regularized mask as torch tensor
        """
        # RT: mild closing to straighten boundaries
        rt = closing(mask, self.kernel_rt)
        return rt
    
    def _apply_rr(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply RR regularization (opening then closing).
        
        Args:
            mask: [B,1,H,W] torch tensor on GPU
            
        Returns:
            RR-regularized mask as torch tensor
        """
        # RR: opening then closing to remove noise and fill small gaps
        rr_tmp = opening(mask, self.kernel_rr)
        rr = closing(rr_tmp, self.kernel_rr)
        return rr
    
    def _apply_fer(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply FER regularization (edge-aware dilation).
        
        Args:
            mask: [B,1,H,W] torch tensor on GPU
            
        Returns:
            FER-regularized mask as torch tensor
        """
        # FER: edge-aware dilation then threshold
        # Convert mask to 0-255 range for edge detection
        mask_255 = mask * 255.0
        
        # Apply Canny edge detection
        edges = canny(mask_255, low_threshold=50.0, high_threshold=150.0)[0]
        
        # Dilate edges
        dilated_edges = dilation(edges, self.kernel_edge)
        
        # Combine dilated edges with original mask
        fer = ((dilated_edges > 0) | (mask > 0.5)).float()
        
        return fer

    def apply_batch(self, masks: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """Process a batch of masks using GPU acceleration.
        
        Args:
            masks: List of numpy masks to regularize
            
        Returns:
            Dictionary with lists of regularized masks for each type
        """
        # Convert numpy arrays to torch tensors and move to GPU
        mask_batch = []
        for mask in masks:
            # Ensure binary float mask
            m = (mask > 0.5).astype(np.float32)
            # Add channel dimension and convert to tensor
            tensor_mask = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
            mask_batch.append(tensor_mask)
        
        # Stack masks into a single batch tensor
        if len(mask_batch) > 0:
            mask_batch_tensor = torch.cat(mask_batch, dim=0).to(self.device)
            
            # Apply regularizations
            results = self.apply(mask_batch_tensor)
            
            # Convert results back to numpy for compatibility
            numpy_results = {}
            for reg_type, tensor_result in results.items():
                numpy_results[reg_type] = [
                    tensor_result[i, 0].cpu().numpy() 
                    for i in range(tensor_result.shape[0])
                ]
            
            return numpy_results
        else:
            return {"original": [], "rt": [], "rr": [], "fer": []}
            
    def cpu_fallback(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """CPU fallback method when GPU is not available.
        
        This implementation matches the original HybridRegularizer.apply()
        but will only be used if no GPU is available.
        """
        # Ensure binary float mask
        m = (mask > 0.5).astype(np.float32)

        # RT: mild closing to straighten boundaries a bit
        kernel_rt = np.ones((3, 3), np.uint8)
        rt = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel_rt, iterations=1).astype(np.float32)

        # RR: opening then closing to remove noise and fill small gaps
        kernel_rr = np.ones((5, 5), np.uint8)
        rr_tmp = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kernel_rr, iterations=1)
        rr = cv2.morphologyEx(rr_tmp, cv2.MORPH_CLOSE, kernel_rr, iterations=1).astype(np.float32)

        # FER: edge-aware dilation then threshold
        edges = cv2.Canny((m * 255).astype(np.uint8), 50, 150)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        fer = ((dilated > 0) | (m > 0.5)).astype(np.float32)

        return {
            "original": m,
            "rt": rt,
            "rr": rr,
            "fer": fer,
        }