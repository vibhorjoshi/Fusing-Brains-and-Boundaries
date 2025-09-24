"""
GPU-Accelerated Mask R-CNN Implementation for Building Footprint Extraction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from tqdm import tqdm

class BuildingDatasetGPU(Dataset):
    """Dataset wrapper for building footprint patches optimized for GPU."""
    
    def __init__(self, patches: List[np.ndarray], masks: List[np.ndarray], 
                 transform=None, device="cuda"):
        self.patches = patches
        self.masks = masks
        self.transform = transform
        self.device = device
        assert len(self.patches) == len(self.masks)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int):
        x = self.patches[idx].astype(np.float32)
        y = (self.masks[idx] > 0).astype(np.uint8)
        
        # Convert to binary mask
        binary_mask = (y > 0).astype(np.uint8)
        
        # Create bounding box from mask
        pos = np.where(binary_mask)
        if len(pos[0]) == 0:  # Empty mask
            # Default small box in corner if no building
            xmin, ymin, xmax, ymax = 0, 0, 10, 10
        else:
            xmin, ymin = np.min(pos[1]), np.min(pos[0])
            xmax, ymax = np.max(pos[1]), np.max(pos[0])
        
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)  # Building class = 1
        masks = torch.from_numpy(binary_mask).unsqueeze(0)
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels, 
            'masks': masks
        }
        
        # Apply transformations if available
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)
            
        return x, target


class GPUMaskRCNNTrainer:
    """GPU-accelerated Mask R-CNN trainer with mixed precision support."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = GradScaler() if hasattr(torch.cuda, 'amp') else None
        self.use_amp = hasattr(torch.cuda, 'amp') and self.config.get("USE_MIXED_PRECISION", False)
        
    def create_model(self, num_classes: int = 2, pretrained: bool = True):
        """Create Mask R-CNN model with GPU optimizations."""
        # Use pretrained weights for faster convergence
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(weights=weights)
        
        # Replace box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            
        model = model.to(self.device)
        self.model = model
        return model
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50) -> Tuple[List[float], List[float]]:
        """Train Mask R-CNN with GPU acceleration and mixed precision."""
        
        if self.model is None:
            self.create_model()
            
        # Optimizer with weight decay
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.config.get("GPU_LEARNING_RATE", 2e-4), 
                               weight_decay=self.config.get("GPU_WEIGHT_DECAY", 1e-4))
        
        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, 
                                                          factor=0.5, verbose=True)
        
        train_losses = []
        val_ious = []
        best_val_iou = 0.0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.model.train()
            epoch_loss = 0.0
            batch_losses = []
            
            # Training loop with progress bar
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for images, targets in train_bar:
                # Move data to GPU
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision if available
                if self.use_amp:
                    with autocast():
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    # Scale loss and backward pass
                    self.scaler.scale(losses).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Standard full-precision training
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()
                
                batch_loss = float(losses.item())
                batch_losses.append(batch_loss)
                epoch_loss += batch_loss
                
                # Update progress bar
                train_bar.set_postfix({"loss": batch_loss})
            
            # Compute average loss for the epoch
            avg_loss = epoch_loss / max(1, len(train_loader))
            train_losses.append(avg_loss)
            
            # Validation
            self.model.eval()
            val_iou = self.evaluate(val_loader)
            val_ious.append(val_iou)
            
            # Update learning rate based on validation performance
            lr_scheduler.step(val_iou)
            
            # Save best model
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                if hasattr(self.model, "module"):  # For DataParallel
                    torch.save(self.model.module.state_dict(), "outputs/models/maskrcnn_best.pth")
                else:
                    torch.save(self.model.state_dict(), "outputs/models/maskrcnn_best.pth")
            
            # Report time and metrics
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val IoU: {val_iou:.4f} | Time: {epoch_time:.1f}s")
        
        return train_losses, val_ious
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set."""
        ious = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                
                # Get predictions
                outputs = self.model(images)
                
                # Compare with targets
                for i, (out, target) in enumerate(zip(outputs, targets)):
                    # Skip if no masks predicted
                    if len(out.get("masks", [])) == 0:
                        ious.append(0.0)
                        continue
                    
                    # Get highest confidence mask
                    pred = (out["masks"][0, 0] > 0.5).cpu().numpy()
                    gt = target["masks"][0].cpu().numpy()
                    
                    # Compute IoU
                    inter = np.logical_and(pred, gt).sum()
                    union = np.logical_or(pred, gt).sum()
                    iou = inter / (union + 1e-8) if union > 0 else 0.0
                    ious.append(float(iou))
        
        # Return mean IoU
        return sum(ious) / max(1, len(ious))


class GPUMaskRCNNInference:
    """GPU-accelerated inference for Mask R-CNN with batched processing."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def process_batch(self, patches: List[np.ndarray], 
                     score_threshold: float = 0.7,
                     mask_threshold: float = 0.5) -> Tuple[List[np.ndarray], List]:
        """
        Process a batch of patches using GPU acceleration.
        
        Args:
            patches: List of image patches (CxHxW)
            score_threshold: Confidence threshold for detections
            mask_threshold: Threshold for binary mask creation
            
        Returns:
            Tuple of (masks_list, predictions_list)
        """
        # Convert to tensors and move to GPU
        with torch.no_grad():
            batch_tensors = [torch.from_numpy(p).to(self.device) for p in patches]
            
            # Run inference
            outputs = self.model(batch_tensors)
            
            # Process outputs
            all_masks = []
            all_preds = []
            
            for i, out in enumerate(outputs):
                masks = []
                # Filter by score
                keep = out["scores"] > score_threshold
                
                if keep.sum() > 0:
                    filtered_masks = out["masks"][keep]
                    # Convert to binary masks
                    for m in filtered_masks:
                        masks.append((m[0] > mask_threshold).float().cpu().numpy())
                
                all_masks.append(masks)
                all_preds.append(out)
            
            return all_masks, all_preds
    
    def process_patch(self, patch: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
        """Process a single patch - wrapper for compatibility."""
        masks, preds = self.process_batch([patch])
        return masks[0], preds[0]


def create_dataloaders(patches, masks, batch_size=8, val_split=0.2, num_workers=4):
    """Create train and validation dataloaders with GPU optimization."""
    # Split data
    n_val = int(len(patches) * val_split)
    n_train = len(patches) - n_val
    
    # Create datasets
    train_dataset = BuildingDatasetGPU(patches[:n_train], masks[:n_train])
    val_dataset = BuildingDatasetGPU(patches[n_train:], masks[n_train:])
    
    # Create dataloaders with pinned memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader, val_loader