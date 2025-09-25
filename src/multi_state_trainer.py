"""
Multi-State GPU Training Pipeline for Building Footprint Extraction

This module implements a high-performance training pipeline for large-scale
building footprint extraction across multiple US states. The pipeline leverages:

1. GPU-accelerated Mask R-CNN training with mixed precision
2. GPU-optimized regularizers with batch processing
3. Parallelized DQN training for adaptive fusion
4. Multi-GPU support for distributed training
5. Checkpoint management and experiment tracking
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
from datetime import datetime
import argparse

# Import GPU-accelerated components
from src.gpu_trainer import GPUMaskRCNNTrainer, BuildingDatasetGPU
from src.gpu_regularizer import GPURegularizer
from src.gpu_adaptive_fusion import GPUAdaptiveFusion

# Import other utilities
from src.data_handler import DataHandler
from src.evaluator import IoUEvaluator
from src.config import Config
from src.utils import setup_logger


class MultiStateTrainingPipeline:
    """High-performance training pipeline for multi-state building footprint extraction.
    
    This pipeline enables training on large datasets across multiple states
    using GPU acceleration and distributed training capabilities.
    """
    
    def __init__(self, config_path=None, distributed=False, local_rank=0):
        """Initialize the multi-state training pipeline.
        
        Args:
            config_path: Path to configuration file
            distributed: Whether to use distributed training
            local_rank: Local process rank for distributed training
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Setup distributed training
        self.distributed = distributed
        self.local_rank = local_rank
        self.is_main_process = local_rank == 0
        
        if distributed:
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.data_handler = DataHandler(self.config)
        self.mask_rcnn = GPUMaskRCNNTrainer(self.config)
        self.regularizer = GPURegularizer(self.config)
        self.fusion = GPUAdaptiveFusion(self.config)
        self.evaluator = IoUEvaluator()
        
        # Setup experiment tracking
        self.experiment_name = f"multi_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = os.path.join("outputs", "models", self.experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.results_dir, "training.log")
        self.logger = setup_logger("multi_state_training", log_file, self.is_main_process)
        
        # Track states and metrics
        self.states = []
        self.metrics = {}
        
    def setup_distributed(self):
        """Initialize distributed training if enabled."""
        if self.distributed:
            self.logger.info(f"Initializing distributed training on rank {self.local_rank}")
            dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.logger.info(f"World size: {self.world_size}")
        else:
            self.world_size = 1
    
    def load_state_data(self, state_list: List[str], max_samples_per_state: int = None):
        """Load and prepare data from multiple states.
        
        Args:
            state_list: List of state names to load
            max_samples_per_state: Maximum samples to load per state
        
        Returns:
            Combined dataset with samples from all states
        """
        self.logger.info(f"Loading data from {len(state_list)} states")
        
        all_patches, all_masks = [], []
        state_samples = {}
        
        for state in state_list:
            self.logger.info(f"Loading {state} data...")
            state_data = self.data_handler.load_state_data(state)
            
            if state_data and "patches" in state_data:
                patches = state_data["patches"]
                masks = state_data["masks"]
                
                if max_samples_per_state and len(patches) > max_samples_per_state:
                    # Sample randomly if too many samples
                    indices = np.random.choice(
                        len(patches), max_samples_per_state, replace=False)
                    patches = [patches[i] for i in indices]
                    masks = [masks[i] for i in indices]
                
                self.logger.info(f"  - {state}: {len(patches)} samples")
                state_samples[state] = len(patches)
                
                all_patches.extend(patches)
                all_masks.extend(masks)
        
        self.states = state_list
        self.state_samples = state_samples
        
        if not all_patches:
            self.logger.error("No data loaded! Check state names and data paths.")
            return None
        
        self.logger.info(f"Total dataset size: {len(all_patches)} samples")
        return BuildingDatasetGPU(all_patches, all_masks, device=self.device)
    
    def create_dataloaders(self, dataset, train_ratio=0.8, batch_size=8):
        """Create training and validation dataloaders.
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio of training samples
            batch_size: Batch size for dataloaders
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Split dataset
        dataset_size = len(dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])
        
        self.logger.info(f"Training set: {train_size} samples")
        self.logger.info(f"Validation set: {val_size} samples")
        
        # Create dataloaders
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=self.config.DATALOADER_NUM_WORKERS,
                pin_memory=True
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=self.config.DATALOADER_NUM_WORKERS,
                pin_memory=True
            )
        else:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config.DATALOADER_NUM_WORKERS,
                pin_memory=True
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.DATALOADER_NUM_WORKERS,
                pin_memory=True
            )
        
        return train_dataloader, val_dataloader
    
    def train_mask_rcnn(self, train_dataloader, val_dataloader, num_epochs=50):
        """Train Mask R-CNN model with GPU acceleration.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Trained model and training metrics
        """
        self.logger.info(f"Training Mask R-CNN for {num_epochs} epochs")
        
        # Create model
        self.mask_rcnn.create_model(num_classes=2, pretrained=True)
        model = self.mask_rcnn.model
        
        if self.distributed:
            model = DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank)
        
        # Setup optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, 
            lr=self.config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        # Setup learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.LR_STEP_SIZE,
            gamma=self.config.LR_GAMMA
        )
        
        # Setup mixed precision training
        scaler = GradScaler() if self.mask_rcnn.use_amp else None
        
        # Training metrics
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "epoch_times": []
        }
        
        best_val_loss = float("inf")
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Set train mode
            model.train()
            
            # Reset sampler for distributed training
            if self.distributed:
                train_dataloader.sampler.set_epoch(epoch)
            
            # Training metrics for current epoch
            train_loss = 0
            train_iter = 0
            
            # Progress bar
            if self.is_main_process:
                pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = train_dataloader
            
            # Training step
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Zero gradients
                if optimizer is not None: optimizer.zero_grad()
                
                if self.mask_rcnn.use_amp:
                    # Mixed precision forward pass
                    with autocast():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    # Scale and backpropagate
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward pass
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Backpropagate
                    losses.backward()
                    optimizer.step()
                
                # Update metrics
                train_loss += losses.item()
                train_iter += 1
                
                # Update progress bar
                if self.is_main_process:
                    pbar.set_postfix({"loss": losses.item()})
            
            # Update learning rate
            lr_scheduler.step()
            
            # Average training loss
            train_loss /= train_iter
            
            # Validation step
            val_loss = self.validate_mask_rcnn(model, val_dataloader)
            
            # Record metrics
            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["epoch_times"].append(time.time() - epoch_start)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Time: {metrics['epoch_times'][-1]:.2f}s"
            )
            
            # Save checkpoints
            if self.is_main_process:
                # Save latest model
                torch.save(
                    model.state_dict(),
                    os.path.join(self.results_dir, f"mask_rcnn_latest.pth")
                )
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.results_dir, f"mask_rcnn_best.pth")
                    )
                    self.logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
        
        # Save final metrics
        if self.is_main_process:
            with open(os.path.join(self.results_dir, "mask_rcnn_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        
        self.metrics["mask_rcnn"] = metrics
        return model, metrics
    
    def validate_mask_rcnn(self, model, dataloader):
        """Run validation for Mask R-CNN.
        
        Args:
            model: Model to validate
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        model.eval()
        val_loss = 0
        val_iter = 0
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                if self.mask_rcnn.use_amp:
                    with autocast():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                else:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                # Update metrics
                val_loss += losses.item()
                val_iter += 1
        
        # Average validation loss
        val_loss /= val_iter
        return val_loss
    
    def train_adaptive_fusion(self, train_dataloader, val_dataloader, num_epochs=30):
        """Train the adaptive fusion model with GPU acceleration.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Trained fusion model and training metrics
        """
        self.logger.info(f"Training Adaptive Fusion for {num_epochs} epochs")
        
        # Load the best Mask R-CNN model
        best_model_path = os.path.join(self.results_dir, "mask_rcnn_best.pth")
        if os.path.exists(best_model_path):
            self.mask_rcnn.model.load_state_dict(torch.load(best_model_path))
        
        model = self.mask_rcnn.model.eval()  # Set to evaluation mode
        
        # Training metrics
        metrics = {
            "train_reward": [],
            "val_reward": [],
            "train_loss": [],
            "epoch_times": []
        }
        
        best_val_reward = float("-inf")
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Reset sampler for distributed training
            if self.distributed:
                train_dataloader.sampler.set_epoch(epoch)
            
            # Training metrics for current epoch
            train_rewards = []
            train_losses = []
            
            # Progress bar
            if self.is_main_process:
                pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = train_dataloader
            
            # Training step
            for images, targets in pbar:
                # Get mask predictions from Mask R-CNN
                with torch.no_grad():
                    images = [img.to(self.device) for img in images]
                    predictions = model(images)
                
                # Extract predicted masks
                batch_masks = []
                gt_masks = []
                
                for i, pred in enumerate(predictions):
                    # Get predicted masks
                    masks = pred["masks"]
                    scores = pred["scores"]
                    
                    if len(scores) > 0 and scores[0] > self.config.DETECTION_THRESHOLD:
                        # Use the mask with highest confidence
                        batch_masks.append(masks[0, 0].cpu().numpy())
                    else:
                        # Use empty mask
                        batch_masks.append(np.zeros((images[0].shape[1], images[0].shape[2])))
                    
                    # Get ground truth mask
                    gt_masks.append(targets[i]["masks"][0, 0].cpu().numpy())
                
                # Apply regularizers to predicted masks
                reg_outputs = self.regularizer.apply_batch(batch_masks)
                
                # Train adaptive fusion on batch
                fused_masks, rewards = self.fusion.process_batch(
                    reg_outputs, gt_masks, training=True)
                
                # Perform DQN update
                loss = self.fusion.train_step()
                
                if loss is not None:
                    train_losses.append(loss)
                
                # Record rewards
                train_rewards.extend(rewards)
                
                # Update target network periodically
                if epoch % self.config.RL_TARGET_UPDATE_FREQ == 0:
                    self.fusion.update_target_network()
                
                # Update exploration rate
                self.fusion.decay_epsilon()
                
                # Update progress bar
                if self.is_main_process and loss is not None:
                    pbar.set_postfix({
                        "loss": loss,
                        "reward": np.mean(rewards),
                        "epsilon": self.fusion.epsilon
                    })
            
            # Calculate average metrics
            avg_train_reward = np.mean(train_rewards) if train_rewards else 0
            avg_train_loss = np.mean(train_losses) if train_losses else 0
            
            # Validation step
            val_reward = self.validate_fusion(val_dataloader)
            
            # Record metrics
            metrics["train_reward"].append(avg_train_reward)
            metrics["val_reward"].append(val_reward)
            metrics["train_loss"].append(avg_train_loss)
            metrics["epoch_times"].append(time.time() - epoch_start)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Reward: {avg_train_reward:.4f}, "
                f"Val Reward: {val_reward:.4f}, "
                f"Loss: {avg_train_loss:.4f}, "
                f"Epsilon: {self.fusion.epsilon:.4f}, "
                f"Time: {metrics['epoch_times'][-1]:.2f}s"
            )
            
            # Save checkpoints
            if self.is_main_process:
                # Save latest model
                self.fusion.save_model(
                    os.path.join(self.results_dir, "fusion_latest.pth"))
                
                # Save best model
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    self.fusion.save_model(
                        os.path.join(self.results_dir, "fusion_best.pth"))
                    self.logger.info(f"New best fusion model saved (reward: {val_reward:.4f})")
        
        # Save final metrics
        if self.is_main_process:
            with open(os.path.join(self.results_dir, "fusion_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        
        self.metrics["fusion"] = metrics
        return self.fusion, metrics
    
    def validate_fusion(self, dataloader):
        """Run validation for adaptive fusion.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation reward
        """
        model = self.mask_rcnn.model.eval()
        val_rewards = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                # Get mask predictions from Mask R-CNN
                images = [img.to(self.device) for img in images]
                predictions = model(images)
                
                # Extract predicted masks
                batch_masks = []
                gt_masks = []
                
                for i, pred in enumerate(predictions):
                    # Get predicted masks
                    masks = pred["masks"]
                    scores = pred["scores"]
                    
                    if len(scores) > 0 and scores[0] > self.config.DETECTION_THRESHOLD:
                        # Use the mask with highest confidence
                        batch_masks.append(masks[0, 0].cpu().numpy())
                    else:
                        # Use empty mask
                        batch_masks.append(np.zeros((images[0].shape[1], images[0].shape[2])))
                    
                    # Get ground truth mask
                    gt_masks.append(targets[i]["masks"][0, 0].cpu().numpy())
                
                # Apply regularizers to predicted masks
                reg_outputs = self.regularizer.apply_batch(batch_masks)
                
                # Apply adaptive fusion (no training)
                _, rewards = self.fusion.process_batch(
                    reg_outputs, gt_masks, training=False)
                
                # Record rewards
                val_rewards.extend(rewards)
        
        # Average validation reward
        avg_val_reward = np.mean(val_rewards) if val_rewards else 0
        return avg_val_reward
    
    def evaluate_on_states(self, state_list=None, max_samples=100):
        """Evaluate models on specific states or all loaded states.
        
        Args:
            state_list: List of states to evaluate (defaults to all loaded states)
            max_samples: Maximum samples per state for evaluation
            
        Returns:
            Dictionary of evaluation metrics per state
        """
        self.logger.info("Running evaluation on states")
        
        if state_list is None:
            state_list = self.states
        
        # Load best models
        mask_rcnn_path = os.path.join(self.results_dir, "mask_rcnn_best.pth")
        fusion_path = os.path.join(self.results_dir, "fusion_best.pth")
        
        if os.path.exists(mask_rcnn_path):
            self.mask_rcnn.model.load_state_dict(torch.load(mask_rcnn_path))
        if os.path.exists(fusion_path):
            self.fusion.load_model(fusion_path)
            
        model = self.mask_rcnn.model.eval()
        
        # Store metrics per state
        state_metrics = {}
        
        for state in state_list:
            self.logger.info(f"Evaluating on {state}")
            
            # Load state data
            state_data = self.data_handler.load_state_data(state)
            
            if not state_data:
                self.logger.warning(f"No data found for {state}, skipping evaluation")
                continue
                
            patches = state_data["patches"]
            gt_masks = state_data["masks"]
            
            if max_samples and len(patches) > max_samples:
                # Sample randomly if too many samples
                indices = np.random.choice(len(patches), max_samples, replace=False)
                patches = [patches[i] for i in indices]
                gt_masks = [gt_masks[i] for i in indices]
            
            # Create dataset and dataloader
            dataset = BuildingDatasetGPU(patches, gt_masks, device=self.device)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.DATALOADER_NUM_WORKERS,
                pin_memory=True
            )
            
            # Track metrics
            mask_rcnn_iou = []
            rt_iou = []
            rr_iou = []
            fer_iou = []
            fusion_iou = []
            
            with torch.no_grad():
                for images, targets in tqdm(dataloader, desc=f"Evaluating {state}"):
                    # Get mask predictions
                    images = [img.to(self.device) for img in images]
                    predictions = model(images)
                    
                    # Process batch
                    batch_masks = []
                    batch_gt = []
                    
                    for i, pred in enumerate(predictions):
                        # Get predicted masks
                        masks = pred["masks"]
                        scores = pred["scores"]
                        
                        if len(scores) > 0 and scores[0] > self.config.DETECTION_THRESHOLD:
                            # Use the mask with highest confidence
                            mask = masks[0, 0].cpu().numpy()
                        else:
                            # Use empty mask
                            mask = np.zeros((images[0].shape[1], images[0].shape[2]))
                        
                        batch_masks.append(mask)
                        
                        # Get ground truth mask
                        gt = targets[i]["masks"][0, 0].cpu().numpy()
                        batch_gt.append(gt)
                        
                        # Calculate Mask R-CNN IoU
                        mask_iou = self.evaluator.compute_iou(mask, gt)
                        mask_rcnn_iou.append(mask_iou)
                    
                    # Apply regularizers
                    reg_outputs = self.regularizer.apply_batch(batch_masks)
                    
                    # Apply fusion
                    fused_masks, _ = self.fusion.process_batch(
                        reg_outputs, batch_gt, training=False)
                    
                    # Calculate IoUs for regularizers and fusion
                    for i in range(len(batch_masks)):
                        # IoUs for individual regularizers
                        rt_iou.append(self.evaluator.compute_iou(
                            reg_outputs["rt"][i], batch_gt[i]))
                        rr_iou.append(self.evaluator.compute_iou(
                            reg_outputs["rr"][i], batch_gt[i]))
                        fer_iou.append(self.evaluator.compute_iou(
                            reg_outputs["fer"][i], batch_gt[i]))
                        
                        # IoU for fusion
                        fusion_iou.append(self.evaluator.compute_iou(
                            fused_masks[i], batch_gt[i]))
            
            # Calculate average metrics
            state_metrics[state] = {
                "sample_count": len(patches),
                "mask_rcnn_iou": np.mean(mask_rcnn_iou),
                "rt_iou": np.mean(rt_iou),
                "rr_iou": np.mean(rr_iou),
                "fer_iou": np.mean(fer_iou),
                "fusion_iou": np.mean(fusion_iou)
            }
            
            # Log results
            self.logger.info(f"Results for {state} ({len(patches)} samples):")
            self.logger.info(f"  Mask R-CNN IoU: {state_metrics[state]['mask_rcnn_iou']:.4f}")
            self.logger.info(f"  RT IoU: {state_metrics[state]['rt_iou']:.4f}")
            self.logger.info(f"  RR IoU: {state_metrics[state]['rr_iou']:.4f}")
            self.logger.info(f"  FER IoU: {state_metrics[state]['fer_iou']:.4f}")
            self.logger.info(f"  Fusion IoU: {state_metrics[state]['fusion_iou']:.4f}")
        
        # Save evaluation results
        if self.is_main_process:
            with open(os.path.join(self.results_dir, "state_evaluation.json"), "w") as f:
                json.dump(state_metrics, f, indent=2)
        
        self.metrics["state_evaluation"] = state_metrics
        return state_metrics
    
    def generate_visualizations(self):
        """Generate performance visualizations from training metrics."""
        if not self.is_main_process:
            return
            
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        
        # Create output directory
        figures_dir = os.path.join("outputs", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot Mask R-CNN training curves
        if "mask_rcnn" in self.metrics:
            metrics = self.metrics["mask_rcnn"]
            
            plt.figure(figsize=(10, 6))
            plt.plot(metrics["train_loss"], label="Train Loss")
            plt.plot(metrics["val_loss"], label="Validation Loss")
            plt.title("Mask R-CNN Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "mask_rcnn_loss.png"), dpi=150)
            plt.close()
        
        # Plot Adaptive Fusion training curves
        if "fusion" in self.metrics:
            metrics = self.metrics["fusion"]
            
            plt.figure(figsize=(10, 6))
            plt.plot(metrics["train_reward"], label="Train Reward")
            plt.plot(metrics["val_reward"], label="Validation Reward")
            plt.title("Adaptive Fusion Training Reward")
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "fusion_reward.png"), dpi=150)
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot(metrics["train_loss"], label="Training Loss")
            plt.title("Adaptive Fusion Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "fusion_loss.png"), dpi=150)
            plt.close()
        
        # Plot state evaluation results
        if "state_evaluation" in self.metrics:
            state_metrics = self.metrics["state_evaluation"]
            
            # Bar chart comparing methods across states
            states = list(state_metrics.keys())
            mask_rcnn_ious = [state_metrics[s]["mask_rcnn_iou"] for s in states]
            rt_ious = [state_metrics[s]["rt_iou"] for s in states]
            rr_ious = [state_metrics[s]["rr_iou"] for s in states]
            fer_ious = [state_metrics[s]["fer_iou"] for s in states]
            fusion_ious = [state_metrics[s]["fusion_iou"] for s in states]
            
            x = np.arange(len(states))
            width = 0.15
            
            plt.figure(figsize=(12, 8))
            plt.bar(x - 2*width, mask_rcnn_ious, width, label="Mask R-CNN")
            plt.bar(x - width, rt_ious, width, label="RT")
            plt.bar(x, rr_ious, width, label="RR")
            plt.bar(x + width, fer_ious, width, label="FER")
            plt.bar(x + 2*width, fusion_ious, width, label="RL Fusion")
            
            plt.title("IoU by State and Method")
            plt.xlabel("State")
            plt.ylabel("IoU")
            plt.xticks(x, states, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "state_comparison.png"), dpi=150)
            plt.close()
    
    def run_pipeline(self, state_list, num_mask_rcnn_epochs=50, num_fusion_epochs=30):
        """Run the complete training pipeline on multiple states.
        
        Args:
            state_list: List of states to train on
            num_mask_rcnn_epochs: Number of epochs for Mask R-CNN training
            num_fusion_epochs: Number of epochs for fusion training
            
        Returns:
            Dictionary of training and evaluation metrics
        """
        # Initialize distributed training (if enabled)
        if self.distributed:
            self.setup_distributed()
        
        # Log training configuration
        self.logger.info(f"Starting multi-state training on {len(state_list)} states")
        self.logger.info(f"States: {', '.join(state_list)}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Distributed: {self.distributed}")
        if self.distributed:
            self.logger.info(f"World size: {self.world_size}")
            self.logger.info(f"Local rank: {self.local_rank}")
        
        # Load data from all states
        dataset = self.load_state_data(state_list)
        
        if not dataset:
            self.logger.error("Failed to load data. Aborting.")
            return None
        
        # Create dataloaders
        train_dataloader, val_dataloader = self.create_dataloaders(
            dataset, 
            batch_size=self.config.BATCH_SIZE
        )
        
        # Train Mask R-CNN
        _, mask_rcnn_metrics = self.train_mask_rcnn(
            train_dataloader, val_dataloader, num_epochs=num_mask_rcnn_epochs)
        
        # Train adaptive fusion
        _, fusion_metrics = self.train_adaptive_fusion(
            train_dataloader, val_dataloader, num_epochs=num_fusion_epochs)
        
        # Evaluate on individual states
        if self.is_main_process:
            state_metrics = self.evaluate_on_states(state_list)
            self.generate_visualizations()
        
        # Save complete metrics
        if self.is_main_process:
            all_metrics = {
                "mask_rcnn": mask_rcnn_metrics,
                "fusion": fusion_metrics,
                "state_evaluation": self.metrics.get("state_evaluation", {})
            }
            
            with open(os.path.join(self.results_dir, "all_metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=2)
        
        self.logger.info("Multi-state training pipeline completed successfully")
        
        return self.metrics


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-State GPU Training Pipeline")
    
    parser.add_argument("--config", type=str, default=None,
                      help="Path to configuration file")
    parser.add_argument("--states", type=str, nargs="+", required=True,
                      help="List of states to train on")
    parser.add_argument("--mask-rcnn-epochs", type=int, default=50,
                      help="Number of epochs for Mask R-CNN training")
    parser.add_argument("--fusion-epochs", type=int, default=30,
                      help="Number of epochs for fusion training")
    parser.add_argument("--distributed", action="store_true",
                      help="Enable distributed training")
    parser.add_argument("--local-rank", type=int, default=0,
                      help="Local rank for distributed training")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Initialize pipeline
    pipeline = MultiStateTrainingPipeline(
        args.config,
        distributed=args.distributed,
        local_rank=args.local_rank
    )
    
    # Run training pipeline
    pipeline.run_pipeline(
        args.states,
        num_mask_rcnn_epochs=args.mask_rcnn_epochs,
        num_fusion_epochs=args.fusion_epochs
    )