"""
Extended Mask R-CNN Trainer with Pretrained Models and Fine-Tuning Support

This module enhances the base Mask R-CNN trainer with support for:
1. Loading pretrained backbone models (ImageNet, COCO)
2. Advanced fine-tuning strategies (freezing layers, gradual unfreezing)
3. Extended training for more epochs with proper convergence monitoring
4. Optimization techniques like learning rate scheduling and mixed precision
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm


class ExtendedMaskRCNNTrainer:
    """Enhanced Mask R-CNN trainer with pretrained model support and advanced fine-tuning."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.config.get("USE_MIXED_PRECISION", True) else None
        
        # Tracking metrics
        self.train_losses = []
        self.val_metrics = []
        
    def create_model(self, num_classes: int = 2, pretrained_type: str = "coco"):
        """Create a new model with pretrained backbone.
        
        Args:
            num_classes: Number of classes for segmentation (including background)
            pretrained_type: Type of pretraining - 'imagenet', 'coco', or 'none'
        """
        if pretrained_type.lower() == "none":
            # Initialize model with random weights
            model = maskrcnn_resnet50_fpn(weights=None)
        elif pretrained_type.lower() == "imagenet":
            # Initialize with ImageNet weights for backbone only
            model = maskrcnn_resnet50_fpn(weights=None)
            # Load ResNet50 backbone with ImageNet weights
            backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            # Copy weights to Mask R-CNN backbone
            model.backbone.body.load_state_dict(backbone.state_dict(), strict=False)
        else:
            # Default: COCO-pretrained Mask R-CNN
            model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
        
        # Replace classification head for our number of classes
        in_features = getattr(getattr(model.roi_heads.box_predictor, "cls_score", object()), "in_features", 1024)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor head
        in_features_mask = getattr(getattr(model.roi_heads.mask_predictor, "conv5_mask", object()), "in_channels", 256)
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        # Store model and move to device
        self.model = model.to(self.device)
        return self.model
    
    def load_pretrained(self, checkpoint_path: Union[str, Path]) -> bool:
        """Load pretrained model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if loading successful, False otherwise
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False
            
    def freeze_backbone(self):
        """Freeze backbone layers for fine-tuning."""
        # Freeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone layers for full training."""
        # Unfreeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = True
            
    def setup_optimizer(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):
        """Set up optimizer for training.
        
        Args:
            lr: Learning rate (uses config value if None)
            weight_decay: Weight decay factor (uses config value if None)
        """
        if lr is None:
            lr = self.config.LEARNING_RATE
        
        if weight_decay is None:
            weight_decay = self.config.WEIGHT_DECAY
            
        # Only include parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=self.config.NUM_EPOCHS,
            pct_start=0.2,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: Optional[int] = None) -> Tuple[List[float], List[Dict[str, float]]]:
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train (uses config value if None)
            
        Returns:
            Tuple of (train_losses, val_metrics)
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model first.")
            
        if self.optimizer is None:
            self.setup_optimizer()
            
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
            
        # Reset metrics tracking
        self.train_losses = []
        self.val_metrics = []
        
        # Training variables
        best_iou = 0.0
        best_epoch = 0
        best_model_weights = None
        patience_counter = 0
        patience = self.config.get("EARLY_STOPPING_PATIENCE", 10)
        
        # Set model to training mode
        self.model.to(self.device)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # Use tqdm for progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for images, targets in progress_bar:
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Mixed precision training
                if self.scaler is not None:
                    with autocast():
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        
                    # Backward and optimize with gradient scaling
                    if optimizer is not None: 
                        optimizer.zero_grad()
                    self.scaler.scale(losses).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward and optimize
                    if optimizer is not None:
                        optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()
                
                # Track loss
                epoch_loss += losses.item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({"Loss": losses.item()})
            
            # Calculate average epoch loss
            avg_loss = epoch_loss / max(1, batch_count)
            self.train_losses.append(avg_loss)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Learning rate: {current_lr:.6f}")
            
            # Validation phase
            val_metrics = self.evaluate(val_loader)
            self.val_metrics.append(val_metrics)
            
            # Print validation metrics
            print(f"Validation: IoU={val_metrics['iou']:.4f}, F1={val_metrics['f1']:.4f}")
            
            # Check for improvement
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                best_epoch = epoch
                best_model_weights = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint(f"best_model.pth", 
                                     epoch=epoch, 
                                     metrics=val_metrics)
            else:
                patience_counter += 1
                
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.get("CHECKPOINT_FREQUENCY", 5) == 0:
                self._save_checkpoint(f"checkpoint_epoch{epoch+1}.pth", 
                                     epoch=epoch, 
                                     metrics=val_metrics)
                
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        # Restore best model
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            print(f"Restored best model from epoch {best_epoch+1}")
            
        return self.train_losses, self.val_metrics
        
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics = {
            'iou': 0.0, 
            'precision': 0.0, 
            'recall': 0.0, 
            'f1': 0.0
        }
        
        total_samples = 0
        valid_samples = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                # Move data to device
                images = [img.to(self.device) for img in images]
                
                # Get predictions
                outputs = self.model(images)
                
                # Calculate metrics for each image
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    if 'masks' not in output or len(output['masks']) == 0:
                        continue
                        
                    # Get predicted mask with highest confidence
                    pred_mask = (output['masks'][0, 0] > 0.5).cpu().numpy()
                    
                    # Get ground truth mask
                    gt_mask = target['masks'][0].cpu().numpy()
                    
                    # Calculate metrics
                    tp = np.logical_and(pred_mask, gt_mask).sum()
                    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
                    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
                    tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()
                    
                    # IoU
                    iou = tp / (tp + fp + fn + 1e-8)
                    
                    # Precision
                    precision = tp / (tp + fp + 1e-8)
                    
                    # Recall
                    recall = tp / (tp + fn + 1e-8)
                    
                    # F1 score
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    
                    # Accumulate metrics
                    metrics['iou'] += iou
                    metrics['precision'] += precision
                    metrics['recall'] += recall
                    metrics['f1'] += f1
                    valid_samples += 1
                    
                total_samples += len(images)
                
        # Average metrics
        if valid_samples > 0:
            for key in metrics:
                metrics[key] /= valid_samples
                
        return metrics
        
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint_dir = self.config.MODELS_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Create checkpoint
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'epoch': epoch,
            'metrics': metrics
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def get_feature_maps(self, images: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract intermediate feature maps from the model.
        
        Args:
            images: List of input images
            
        Returns:
            Dictionary of feature maps at different levels
        """
        self.model.eval()
        feature_maps = {}
        
        with torch.no_grad():
            # Get backbone features
            images_tensor = torch.stack([img.to(self.device) for img in images])
            
            # Extract backbone features (C2-C5)
            features = self.model.backbone.body(images_tensor)
            
            # Extract FPN features (P2-P6)
            fpn_features = self.model.backbone.fpn(features)
            
            # Store features
            feature_maps.update(fpn_features)
            
        return feature_maps
        
    def get_logits_and_masks(self, images: List[torch.Tensor]) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
        """Get raw logits and masks from model predictions.
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (outputs, raw_mask_logits)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get model predictions
            images_device = [img.to(self.device) for img in images]
            outputs = self.model(images_device)
            
            # Extract raw mask logits before thresholding
            raw_mask_logits = [output['masks'] for output in outputs]
            
        return outputs, raw_mask_logits
        
    def fine_tune(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 initial_lr: float = 1e-5,
                 fine_tune_epochs: int = 5,
                 full_train_epochs: int = 45) -> Tuple[List[float], List[Dict[str, float]]]:
        """Two-stage fine-tuning process:
        1. Fine-tune only heads with frozen backbone
        2. Train full model with lower learning rate
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            initial_lr: Initial learning rate for fine-tuning
            fine_tune_epochs: Number of epochs for fine-tuning heads only
            full_train_epochs: Number of epochs for full model training
            
        Returns:
            Tuple of (train_losses, val_metrics)
        """
        print("=" * 60)
        print("Stage 1: Fine-tuning prediction heads (backbone frozen)")
        print("=" * 60)
        
        # Freeze backbone
        self.freeze_backbone()
        
        # Setup optimizer with low learning rate
        self.setup_optimizer(lr=initial_lr)
        
        # Train for a few epochs
        self.train(train_loader, val_loader, num_epochs=fine_tune_epochs)
        
        print("=" * 60)
        print("Stage 2: Training full model")
        print("=" * 60)
        
        # Unfreeze backbone
        self.unfreeze_backbone()
        
        # Setup optimizer with higher learning rate
        self.setup_optimizer()
        
        # Train full model
        return self.train(train_loader, val_loader, num_epochs=full_train_epochs)