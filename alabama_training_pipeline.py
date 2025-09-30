#!/usr/bin/env python3
"""
Alabama State Building Footprint Training Pipeline
GPU-Accelerated Training with 50 Epochs
Binary Mask Generation and IoU Scoring
Live Visualization Support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Optional
import requests
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import asyncio
import websockets
import threading
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    epoch: int
    loss: float
    iou_score: float
    confidence: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class AlabamaStateDataset(Dataset):
    """Dataset class for Alabama state satellite imagery with building footprints"""
    
    def __init__(self, data_dir: str, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load Alabama state satellite imagery samples"""
        samples = []
        
        # Synthetic Alabama state coordinates for training
        alabama_regions = [
            {"name": "Birmingham", "lat": 33.5186, "lon": -86.8104},
            {"name": "Montgomery", "lat": 32.3668, "lon": -86.2999},
            {"name": "Mobile", "lat": 30.6954, "lon": -88.0399},
            {"name": "Huntsville", "lat": 34.7304, "lon": -86.5861},
            {"name": "Tuscaloosa", "lat": 33.2098, "lon": -87.5692},
            {"name": "Hoover", "lat": 33.4054, "lon": -86.8114},
            {"name": "Auburn", "lat": 32.6010, "lon": -85.4883},
            {"name": "Dothan", "lat": 31.2232, "lon": -85.3905},
            {"name": "Decatur", "lat": 34.6059, "lon": -86.9833},
            {"name": "Madison", "lat": 34.6993, "lon": -86.7483}
        ]
        
        for i, region in enumerate(alabama_regions):
            # Generate synthetic samples for each region
            for j in range(20):  # 20 samples per region
                sample = {
                    'id': f'alabama_{region["name"]}_{j:03d}',
                    'region': region["name"],
                    'lat': region["lat"] + np.random.uniform(-0.1, 0.1),
                    'lon': region["lon"] + np.random.uniform(-0.1, 0.1),
                    'image_path': f'synthetic_image_{i}_{j}.png',
                    'mask_path': f'synthetic_mask_{i}_{j}.png',
                    'building_count': np.random.randint(5, 50),
                    'area_sqm': np.random.uniform(1000, 10000)
                }
                samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} Alabama state samples")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        
        # Generate synthetic satellite image (512x512x3)
        image = self._generate_synthetic_satellite_image(sample)
        
        # Generate corresponding building mask
        mask = self._generate_building_mask(sample, image)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask, sample
    
    def _generate_synthetic_satellite_image(self, sample: Dict) -> np.ndarray:
        """Generate synthetic satellite imagery for Alabama state"""
        # Create base satellite-like image
        image = np.random.randint(80, 120, (512, 512, 3), dtype=np.uint8)
        
        # Add terrain features
        self._add_terrain_features(image, sample)
        
        # Add urban structures
        self._add_urban_structures(image, sample)
        
        return image
    
    def _add_terrain_features(self, image: np.ndarray, sample: Dict):
        """Add terrain features specific to Alabama"""
        # Add vegetation (green areas)
        vegetation_mask = np.random.rand(512, 512) > 0.7
        image[vegetation_mask, 1] = np.clip(image[vegetation_mask, 1] + 40, 0, 255)
        
        # Add water bodies (blue areas)
        water_mask = np.random.rand(512, 512) > 0.9
        image[water_mask, 2] = np.clip(image[water_mask, 2] + 60, 0, 255)
        image[water_mask, :2] = np.maximum(image[water_mask, :2] - 30, 0)
        
        # Add roads (darker lines)
        for _ in range(np.random.randint(3, 8)):
            start_x = np.random.randint(0, 512)
            start_y = np.random.randint(0, 512)
            end_x = np.random.randint(0, 512)
            end_y = np.random.randint(0, 512)
            cv2.line(image, (start_x, start_y), (end_x, end_y), (40, 40, 40), 2)
    
    def _add_urban_structures(self, image: np.ndarray, sample: Dict):
        """Add urban structures and buildings"""
        building_count = sample['building_count']
        
        for _ in range(building_count):
            # Random building position and size
            x = np.random.randint(10, 502)
            y = np.random.randint(10, 502)
            w = np.random.randint(8, 40)
            h = np.random.randint(8, 40)
            
            # Building color (concrete/rooftop colors)
            color = (
                np.random.randint(100, 200),
                np.random.randint(100, 200),
                np.random.randint(100, 200)
            )
            
            cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
            
            # Add shadow effect
            shadow_offset = 2
            cv2.rectangle(image, (x+shadow_offset, y+shadow_offset), 
                         (x+w+shadow_offset, y+h+shadow_offset), 
                         (50, 50, 50), -1)
    
    def _generate_building_mask(self, sample: Dict, image: np.ndarray) -> np.ndarray:
        """Generate binary building footprint mask"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        building_count = sample['building_count']
        
        for _ in range(building_count):
            # Random building position and size
            x = np.random.randint(10, 502)
            y = np.random.randint(10, 502)
            w = np.random.randint(8, 40)
            h = np.random.randint(8, 40)
            
            # Create binary mask for building
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        
        return mask

class AdaptiveFusionUNet(nn.Module):
    """Enhanced U-Net with Adaptive Fusion for building footprint extraction"""
    
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(AdaptiveFusionUNet, self).__init__()
        
        # Encoder
        self.encoder1 = self._conv_block(in_channels, features)
        self.encoder2 = self._conv_block(features, features*2)
        self.encoder3 = self._conv_block(features*2, features*4)
        self.encoder4 = self._conv_block(features*4, features*8)
        
        # Bottleneck with attention
        self.bottleneck = self._conv_block(features*8, features*16)
        self.attention = nn.MultiheadAttention(features*16, num_heads=8, batch_first=True)
        
        # Adaptive Fusion Module
        self.fusion_conv = nn.Conv2d(features*16, features*8, kernel_size=1)
        self.fusion_gate = nn.Sigmoid()
        
        # Decoder
        self.decoder4 = self._conv_block(features*16, features*8)
        self.decoder3 = self._conv_block(features*8, features*4)
        self.decoder2 = self._conv_block(features*4, features*2)
        self.decoder1 = self._conv_block(features*2, features)
        
        # Output layer
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck with attention
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Apply attention mechanism
        b, c, h, w = bottleneck.shape
        bottleneck_flat = bottleneck.view(b, c, h*w).permute(0, 2, 1)
        attn_output, _ = self.attention(bottleneck_flat, bottleneck_flat, bottleneck_flat)
        bottleneck_attn = attn_output.permute(0, 2, 1).view(b, c, h, w)
        
        # Adaptive fusion
        fusion_gate = self.fusion_gate(self.fusion_conv(bottleneck_attn))
        bottleneck_fused = bottleneck * fusion_gate + bottleneck_attn * (1 - fusion_gate)
        
        # Decoder path
        dec4 = self.decoder4(torch.cat([self.upconv4(bottleneck_fused), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        
        return torch.sigmoid(self.final_conv(dec1))

class IoULoss(nn.Module):
    """Intersection over Union Loss for binary segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    """Combined BCE and IoU Loss"""
    
    def __init__(self, bce_weight=0.5, iou_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        return self.bce_weight * bce + self.iou_weight * iou

class AlabamaTrainer:
    """GPU-accelerated trainer for Alabama state building footprint extraction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training device: {self.device}")
        
        # Initialize model
        self.model = AdaptiveFusionUNet(
            in_channels=3,
            out_channels=1,
            features=config.get('features', 64)
        ).to(self.device)
        
        # Initialize loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training state
        self.epoch = 0
        self.best_iou = 0.0
        self.training_history = []
        
        # WebSocket for live updates
        self.websocket_server = None
        self.connected_clients = set()
        
    async def start_websocket_server(self):
        """Start WebSocket server for live training updates"""
        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")
            try:
                await websocket.wait_closed()
            finally:
                self.connected_clients.remove(websocket)
                logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")
        
        self.websocket_server = await websockets.serve(handle_client, "localhost", 8765)
        logger.info("WebSocket server started on ws://localhost:8765")
    
    async def broadcast_metrics(self, metrics: TrainingMetrics):
        """Broadcast training metrics to connected clients"""
        if self.connected_clients:
            message = {
                'type': 'training_update',
                'epoch': metrics.epoch,
                'loss': float(metrics.loss),
                'iou_score': float(metrics.iou_score),
                'confidence': float(metrics.confidence),
                'accuracy': float(metrics.accuracy),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1_score': float(metrics.f1_score),
                'timestamp': datetime.now().isoformat()
            }
            
            disconnected = set()
            for client in self.connected_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            for client in disconnected:
                self.connected_clients.remove(client)
    
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict:
        """Calculate comprehensive metrics"""
        pred_binary = (pred > 0.5).float()
        target_binary = target.float()
        
        # IoU calculation
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        iou = intersection / (union + 1e-6)
        
        # Accuracy, Precision, Recall, F1
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum()
        
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        # Confidence (mean prediction probability for positive class)
        confidence = pred[target_binary == 1].mean() if (target_binary == 1).sum() > 0 else torch.tensor(0.0)
        
        return {
            'iou_score': iou.item(),
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1.item(),
            'confidence': confidence.item() if hasattr(confidence, 'item') else float(confidence)
        }
    
    def save_checkpoint(self, metrics: TrainingMetrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': metrics.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'training_history': self.training_history
        }
        
        checkpoint_path = f"checkpoints/alabama_epoch_{metrics.epoch}_iou_{metrics.iou_score:.4f}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    async def train_epoch(self, dataloader: DataLoader) -> TrainingMetrics:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {'iou_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'confidence': 0}
        num_batches = len(dataloader)
        
        for batch_idx, (images, masks, samples) in enumerate(dataloader):
            images = images.to(self.device).float()
            masks = masks.to(self.device).float().unsqueeze(1)  # Add channel dimension
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate metrics
            batch_metrics = self.calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss.item():.4f}, IoU: {batch_metrics['iou_score']:.4f}")
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        metrics = TrainingMetrics(
            epoch=self.epoch,
            loss=avg_loss,
            **avg_metrics
        )
        
        # Update learning rate
        self.scheduler.step(avg_loss)
        
        # Save best model
        if avg_metrics['iou_score'] > self.best_iou:
            self.best_iou = avg_metrics['iou_score']
            self.save_checkpoint(metrics)
        
        self.training_history.append(metrics)
        
        return metrics
    
    async def train(self, epochs: int = 50):
        """Main training loop"""
        logger.info(f"Starting Alabama State training for {epochs} epochs")
        
        # Start WebSocket server
        await self.start_websocket_server()
        
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        
        dataset = AlabamaStateDataset(
            data_dir="./alabama_data",
            transform=transform,
            target_transform=target_transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} samples, {len(dataloader)} batches per epoch")
        
        # Training loop
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            start_time = time.time()
            
            metrics = await self.train_epoch(dataloader)
            
            epoch_time = time.time() - start_time
            
            # Log epoch results
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info(f"Loss: {metrics.loss:.4f}, IoU: {metrics.iou_score:.4f}, "
                       f"Acc: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}")
            
            # Broadcast metrics via WebSocket
            await self.broadcast_metrics(metrics)
            
            # Save visualization every 5 epochs
            if epoch % 5 == 0:
                await self.save_prediction_visualization(dataloader)
        
        logger.info("Training completed!")
        return self.training_history
    
    async def save_prediction_visualization(self, dataloader: DataLoader):
        """Save prediction visualization"""
        self.model.eval()
        
        with torch.no_grad():
            # Get first batch for visualization
            images, masks, samples = next(iter(dataloader))
            images = images.to(self.device).float()
            masks = masks.to(self.device).float()
            
            # Get predictions
            predictions = self.model(images)
            predictions = torch.sigmoid(predictions)
            
            # Move to CPU for visualization
            images_cpu = images.cpu()
            masks_cpu = masks.cpu()
            predictions_cpu = predictions.cpu()
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            img = images_cpu[0].permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = torch.clamp(img, 0, 1)
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Ground truth mask
            axes[0, 1].imshow(masks_cpu[0].squeeze(), cmap='gray')
            axes[0, 1].set_title('Ground Truth Mask')
            axes[0, 1].axis('off')
            
            # Prediction
            axes[0, 2].imshow(predictions_cpu[0].squeeze(), cmap='gray')
            axes[0, 2].set_title('Prediction')
            axes[0, 2].axis('off')
            
            # Binary prediction
            binary_pred = (predictions_cpu[0].squeeze() > 0.5).float()
            axes[1, 0].imshow(binary_pred, cmap='gray')
            axes[1, 0].set_title('Binary Prediction')
            axes[1, 0].axis('off')
            
            # Overlay
            overlay = img.clone()
            mask_overlay = masks_cpu[0].squeeze().unsqueeze(2).repeat(1, 1, 3)
            overlay = overlay * 0.7 + mask_overlay * 0.3 * torch.tensor([0, 1, 0])
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Ground Truth Overlay')
            axes[1, 1].axis('off')
            
            # Prediction overlay
            pred_overlay = img.clone()
            pred_mask_overlay = binary_pred.unsqueeze(2).repeat(1, 1, 3)
            pred_overlay = pred_overlay * 0.7 + pred_mask_overlay * 0.3 * torch.tensor([1, 0, 0])
            axes[1, 2].imshow(pred_overlay)
            axes[1, 2].set_title('Prediction Overlay')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            os.makedirs("visualizations", exist_ok=True)
            plt.savefig(f'visualizations/alabama_epoch_{self.epoch}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved for epoch {self.epoch}")

def main():
    """Main training function"""
    # Training configuration
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 8,
        'num_workers': 4,
        'features': 64
    }
    
    # Initialize trainer
    trainer = AlabamaTrainer(config)
    
    # Run training
    try:
        asyncio.run(trainer.train(epochs=50))
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()