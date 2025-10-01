#!/usr/bin/env python3
"""
Enhanced Building Footprint Pipeline Runner

This script implements the enhanced building footprint extraction pipeline with
all improvements:

1. Fused learned proposals with Mask R-CNN logits as additional streams
2. Enriched RL state with image-conditioned features and CNN embeddings
3. Continuous action space with policy gradient methods (PPO)
4. Increased sample size across multiple states
5. Extended Mask R-CNN training with pre-trained model loading
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import Config
from src.data_handler import RasterDataLoader, BuildingDataset
from src.multi_state_trainer import MultiStateTrainer
from src.extended_maskrcnn import ExtendedMaskRCNNTrainer
from src.gpu_regularizer import GPURegularizer
from src.enhanced_adaptive_fusion import EnhancedAdaptiveFusion
from src.post_processor import PostProcessor
from src.evaluator import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Building Footprint Pipeline')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'inference'],
                        help='Pipeline mode')
    parser.add_argument('--states', type=str, nargs='+',
                        default=None, 
                        help='States to process (default: config.TRAINING_STATES)')
    parser.add_argument('--patches-per-state', type=int, default=None,
                        help='Number of patches per state (default: config.PATCHES_PER_STATE)')
    parser.add_argument('--pretrained', type=str, default='coco',
                        choices=['none', 'imagenet', 'coco', 'custom'],
                        help='Pretrained model type')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to custom pretrained model')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='Number of training epochs (default: config.NUM_EPOCHS)')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Use two-stage fine-tuning approach')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: config.OUTPUT_DIR)')
    
    return parser.parse_args()


def setup_output_dirs(config: Config):
    """Create output directories if they don't exist."""
    for directory in [config.OUTPUT_DIR, config.FIGURES_DIR, 
                      config.MODELS_DIR, config.LOGS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def train_pipeline(config: Config, args):
    """Run the full training pipeline with enhancements."""
    print("=" * 80)
    print("ENHANCED BUILDING FOOTPRINT EXTRACTION PIPELINE")
    print("=" * 80)
    
    # 1. Setup Multi-State Training
    print("\n1. Setting up multi-state training...")
    multi_trainer = MultiStateTrainer(config)
    
    # Load specified states or default from config
    states = args.states or config.TRAINING_STATES
    patches_per_state = args.patches_per_state or config.PATCHES_PER_STATE
    
    loaded_states = multi_trainer.load_states(states, patches_per_state)
    if not loaded_states:
        print("Error: Failed to load any states. Exiting.")
        return
    
    print(f"Successfully loaded states: {', '.join(loaded_states)}")
    
    # 2. Train/Load Mask R-CNN Model
    print("\n2. Setting up extended Mask R-CNN model...")
    mask_rcnn_trainer = ExtendedMaskRCNNTrainer(config)
    
    # Create model with specified pretrained weights
    if args.pretrained == 'custom' and args.model_path:
        print(f"Loading custom pretrained model from: {args.model_path}")
        mask_rcnn_trainer.create_model(num_classes=2, pretrained_type='none')
        mask_rcnn_trainer.load_pretrained(args.model_path)
    else:
        print(f"Creating model with {args.pretrained} pretrained weights")
        mask_rcnn_trainer.create_model(num_classes=2, pretrained_type=args.pretrained)
    
    # Create train and validation data loaders
    train_loader, val_loader, split_info = multi_trainer.data_manager.create_train_val_split()
    print(f"Created data loaders with {split_info['total']['train_size']} training and "
          f"{split_info['total']['val_size']} validation samples")
    
    # Train Mask R-CNN with specified parameters
    num_epochs = args.num_epochs or config.NUM_EPOCHS
    
    if args.fine_tune:
        print("\nUsing two-stage fine-tuning approach...")
        train_losses, val_metrics = mask_rcnn_trainer.fine_tune(
            train_loader, val_loader,
            initial_lr=1e-5,
            fine_tune_epochs=5,
            full_train_epochs=num_epochs - 5
        )
    else:
        print(f"\nTraining model for {num_epochs} epochs...")
        train_losses, val_metrics = mask_rcnn_trainer.train(
            train_loader, val_loader, num_epochs
        )
    
    # Save training visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    val_ious = [metrics['iou'] for metrics in val_metrics]
    ax2.plot(epochs, val_ious, 'r-')
    ax2.set_title('Validation IoU')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('IoU')
    ax2.grid(True)
    
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, 'maskrcnn_training.png'), 
                dpi=config.SAVE_FIGURE_DPI)
    
    # 3. Set up enhanced regularizers
    print("\n3. Setting up GPU-accelerated regularizers...")
    regularizer = GPURegularizer(config)
    
    # 4. Set up enhanced adaptive fusion
    print("\n4. Setting up enhanced adaptive fusion...")
    fusion = EnhancedAdaptiveFusion(config)
    
    # 5. Training and evaluation pipeline
    print("\n5. Running enhanced training pipeline...")
    
    # Get validation samples for visual evaluation
    val_samples = []
    val_ground_truth = []
    
    for images, targets in val_loader:
        # Just get a few samples
        for i in range(min(5, len(images))):
            val_samples.append(images[i].cpu().numpy())
            val_ground_truth.append(targets[i]['masks'][0].cpu().numpy())
        break
    
    # Process samples through the pipeline
    processed_results = []
    
    for i, (image, gt) in enumerate(zip(val_samples, val_ground_truth)):
        print(f"Processing validation sample {i+1}/{len(val_samples)}...")
        
        # Inference with Mask R-CNN
        device = next(mask_rcnn_trainer.model.parameters()).device
        image_tensor = torch.from_numpy(image).to(device)
        outputs, mask_logits = mask_rcnn_trainer.get_logits_and_masks([image_tensor])
        
        # Get prediction and logits
        if len(outputs[0]['masks']) > 0:
            rcnn_mask = outputs[0]['masks'][0, 0].cpu().numpy()
            rcnn_logits = mask_logits[0][0, 0].cpu().numpy()
        else:
            rcnn_mask = np.zeros_like(gt)
            rcnn_logits = np.zeros_like(gt)
        
        # Apply regularizers
        image_tensor_batch = image_tensor.unsqueeze(0)  # Add batch dimension
        mask_tensor = torch.from_numpy(rcnn_mask).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
        
        reg_outputs = {
            'original': image_tensor_batch,
            'proposal': mask_tensor
        }
        
        # Apply each regularizer
        reg_outputs['rt'] = regularizer.apply_rt(mask_tensor)
        reg_outputs['rr'] = regularizer.apply_rr(mask_tensor)
        reg_outputs['fer'] = regularizer.apply_fer(mask_tensor, image_tensor_batch)
        
        # Create ground truth tensor
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
        
        # Extract features for RL state
        state = fusion.extract_features(reg_outputs)
        
        # Select action (fusion weights)
        actions, _, _ = fusion.select_action(state, training=False)
        
        # Apply fusion
        fused_mask = fusion.fuse_masks(reg_outputs, actions)
        fused_numpy = fused_mask[0, 0].cpu().numpy()
        
        # Calculate metrics
        reward = fusion.compute_reward(fused_mask, gt_tensor)
        
        # Post-process if needed
        
        # Store results
        processed_results.append({
            'image': image,
            'ground_truth': gt,
            'rcnn_mask': rcnn_mask,
            'rcnn_logits': rcnn_logits,
            'rt_mask': reg_outputs['rt'][0, 0].cpu().numpy(),
            'rr_mask': reg_outputs['rr'][0, 0].cpu().numpy(),
            'fer_mask': reg_outputs['fer'][0, 0].cpu().numpy(),
            'fused_mask': fused_numpy,
            'reward': reward[0].cpu().item(),
            'fusion_weights': actions[0].cpu().numpy()
        })
    
    # 6. Visualize results
    print("\n6. Generating visualizations...")
    
    for i, result in enumerate(processed_results):
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        
        # Input image
        axs[0, 0].imshow(np.transpose(result['image'], (1, 2, 0)))
        axs[0, 0].set_title('Input Image')
        axs[0, 0].axis('off')
        
        # Ground truth
        axs[0, 1].imshow(result['ground_truth'], cmap='gray')
        axs[0, 1].set_title('Ground Truth')
        axs[0, 1].axis('off')
        
        # Mask R-CNN prediction
        axs[0, 2].imshow(result['rcnn_mask'], cmap='gray')
        axs[0, 2].set_title('Mask R-CNN')
        axs[0, 2].axis('off')
        
        # Mask R-CNN logits
        axs[0, 3].imshow(result['rcnn_logits'], cmap='viridis')
        axs[0, 3].set_title('Mask R-CNN Logits')
        axs[0, 3].axis('off')
        
        # RT regularization
        axs[1, 0].imshow(result['rt_mask'], cmap='gray')
        axs[1, 0].set_title('RT Regularized')
        axs[1, 0].axis('off')
        
        # RR regularization
        axs[1, 1].imshow(result['rr_mask'], cmap='gray')
        axs[1, 1].set_title('RR Regularized')
        axs[1, 1].axis('off')
        
        # FER regularization
        axs[1, 2].imshow(result['fer_mask'], cmap='gray')
        axs[1, 2].set_title('FER Regularized')
        axs[1, 2].axis('off')
        
        # Fused result
        axs[1, 3].imshow(result['fused_mask'], cmap='gray')
        weights = result['fusion_weights']
        weight_text = f"RT: {weights[0]:.2f}, RR: {weights[1]:.2f}, FER: {weights[2]:.2f}, Prop: {weights[3]:.2f}"
        axs[1, 3].set_title(f'Fused Result (IoU: {result["reward"]/100:.3f})\n{weight_text}')
        axs[1, 3].axis('off')
        
        fig.tight_layout()
        fig.savefig(os.path.join(config.FIGURES_DIR, f'enhanced_result_{i+1}.png'), 
                   dpi=config.SAVE_FIGURE_DPI)
        plt.close(fig)
    
    print("\nTraining pipeline completed successfully!")
    print(f"Results saved to: {config.OUTPUT_DIR}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = Config()
    
    # Override output directory if specified
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
        config.FIGURES_DIR = config.OUTPUT_DIR / "figures"
        config.MODELS_DIR = config.OUTPUT_DIR / "models"
        config.LOGS_DIR = config.OUTPUT_DIR / "logs"
    
    # Create output directories
    setup_output_dirs(config)
    
    # Run pipeline based on mode
    if args.mode == 'train':
        train_pipeline(config, args)
    elif args.mode == 'evaluate':
        print("Evaluation mode not implemented yet.")
    elif args.mode == 'inference':
        print("Inference mode not implemented yet.")
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()