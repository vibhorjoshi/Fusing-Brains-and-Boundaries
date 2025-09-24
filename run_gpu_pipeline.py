"""
GPU-Accelerated Building Footprint Extraction Pipeline Runner

This script serves as the main entry point for executing the complete
GPU-accelerated building footprint extraction pipeline, including:

1. Training Mask R-CNN on GPU
2. Processing with GPU-accelerated regularizers
3. Training and applying adaptive fusion with GPU
4. Evaluating performance with benchmarking tools

Example usage:
python run_gpu_pipeline.py --mode train --states Alabama Arizona --epochs 50
python run_gpu_pipeline.py --mode benchmark
python run_gpu_pipeline.py --mode evaluate --states Alabama --max-samples 100
"""

import argparse
import os
import torch
import time
from datetime import datetime

# Import GPU components
from src.gpu_trainer import GPUMaskRCNNTrainer, BuildingDatasetGPU
from src.gpu_regularizer import GPURegularizer
from src.gpu_adaptive_fusion import GPUAdaptiveFusion
from src.benchmarking import PerformanceBenchmark
from src.multi_state_trainer import MultiStateTrainingPipeline

# Import utilities
from src.data_handler import DataHandler
from src.config import Config
from src.utils import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Building Footprint Extraction")
    
    parser.add_argument("--mode", type=str, required=True,
                      choices=["train", "benchmark", "evaluate", "process"],
                      help="Mode to run the pipeline in")
    
    parser.add_argument("--config", type=str, default=None,
                      help="Path to configuration file")
    
    parser.add_argument("--states", type=str, nargs="+",
                      help="List of states to process")
    
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of epochs for training")
    
    parser.add_argument("--fusion-epochs", type=int, default=30,
                      help="Number of epochs for fusion training")
    
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for processing")
    
    parser.add_argument("--max-samples", type=int, default=None,
                      help="Maximum samples per state")
    
    parser.add_argument("--distributed", action="store_true",
                      help="Enable distributed training")
    
    parser.add_argument("--local-rank", type=int, default=0,
                      help="Local rank for distributed training")
    
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Output directory for results")
    
    return parser.parse_args()


def print_gpu_info():
    """Print information about available GPU resources."""
    print("\n=== GPU Information ===")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available. Found {device_count} GPU(s).")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Total Memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"  - CUDA Arch: {props.multi_processor_count} SMs")
            
        # Print current device
        print(f"\nCurrent CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is NOT available. Running on CPU.")
    print("=====================\n")


def train_pipeline(args):
    """Train the complete pipeline on multiple states."""
    # Initialize pipeline
    pipeline = MultiStateTrainingPipeline(
        args.config,
        distributed=args.distributed,
        local_rank=args.local_rank
    )
    
    # Run training
    pipeline.run_pipeline(
        args.states,
        num_mask_rcnn_epochs=args.epochs,
        num_fusion_epochs=args.fusion_epochs
    )
    
    
def run_benchmark(args):
    """Run performance benchmarks comparing CPU and GPU implementations."""
    benchmark = PerformanceBenchmark(args.config)
    benchmark.run_all_benchmarks()
    

def evaluate_model(args):
    """Evaluate trained models on specific states."""
    if not args.states:
        print("Error: Must specify states for evaluation")
        return
    
    # Initialize pipeline
    pipeline = MultiStateTrainingPipeline(
        args.config,
        distributed=False,
        local_rank=0
    )
    
    # Load best models and evaluate on specified states
    state_metrics = pipeline.evaluate_on_states(
        args.states,
        max_samples=args.max_samples
    )
    
    # Generate visualizations
    pipeline.generate_visualizations()
    
    # Print summary
    print("\n=== Evaluation Results ===")
    for state, metrics in state_metrics.items():
        print(f"State: {state} ({metrics['sample_count']} samples)")
        print(f"  - Mask R-CNN IoU: {metrics['mask_rcnn_iou']:.4f}")
        print(f"  - RT IoU: {metrics['rt_iou']:.4f}")
        print(f"  - RR IoU: {metrics['rr_iou']:.4f}")
        print(f"  - FER IoU: {metrics['fer_iou']:.4f}")
        print(f"  - Fusion IoU: {metrics['fusion_iou']:.4f}")
    print("=========================")


def process_states(args):
    """Process states with the full pipeline without training."""
    # Load configuration
    config = Config(args.config)
    
    # Initialize components
    data_handler = DataHandler(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize GPU components
    mask_rcnn = GPUMaskRCNNTrainer(config)
    regularizer = GPURegularizer(config)
    fusion = GPUAdaptiveFusion(config)
    
    # Load best models if available
    model_dir = "outputs/models"
    mask_rcnn_path = os.path.join(model_dir, "mask_rcnn_best.pth")
    fusion_path = os.path.join(model_dir, "fusion_best.pth")
    
    # Create model
    mask_rcnn.create_model(num_classes=2)
    
    if os.path.exists(mask_rcnn_path):
        print(f"Loading Mask R-CNN from {mask_rcnn_path}")
        mask_rcnn.model.load_state_dict(torch.load(mask_rcnn_path))
    
    if os.path.exists(fusion_path):
        print(f"Loading fusion model from {fusion_path}")
        fusion.load_model(fusion_path)
    
    # Set model to evaluation mode
    mask_rcnn.model.eval()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"outputs/processed_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each state
    for state in args.states:
        print(f"\nProcessing {state}...")
        
        # Load state data
        state_data = data_handler.load_state_data(state)
        
        if not state_data:
            print(f"No data found for {state}, skipping")
            continue
            
        patches = state_data["patches"]
        gt_masks = state_data["masks"]
        
        if args.max_samples and len(patches) > args.max_samples:
            # Sample subset if requested
            import numpy as np
            indices = np.random.choice(len(patches), args.max_samples, replace=False)
            patches = [patches[i] for i in indices]
            gt_masks = [gt_masks[i] for i in indices]
        
        print(f"Processing {len(patches)} samples for {state}")
        
        # Create dataset
        dataset = BuildingDatasetGPU(patches, gt_masks, device=device)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Process all batches
        results = {
            "original": [],
            "mask_rcnn": [],
            "rt": [],
            "rr": [],
            "fer": [],
            "fusion": []
        }
        
        with torch.no_grad():
            for images, targets in dataloader:
                # Get mask predictions
                images = [img.to(device) for img in images]
                predictions = mask_rcnn.model(images)
                
                # Extract predicted masks
                batch_masks = []
                batch_gt = []
                
                for i, pred in enumerate(predictions):
                    # Get predicted masks
                    masks = pred["masks"]
                    scores = pred["scores"]
                    
                    if len(scores) > 0 and scores[0] > config.DETECTION_THRESHOLD:
                        # Use the mask with highest confidence
                        mask = masks[0, 0].cpu().numpy()
                    else:
                        # Use empty mask
                        mask = np.zeros((images[0].shape[1], images[0].shape[2]))
                    
                    batch_masks.append(mask)
                    
                    # Store original image and ground truth
                    results["original"].append(images[i].cpu().numpy())
                    results["mask_rcnn"].append(mask)
                    
                    # Store ground truth
                    gt = targets[i]["masks"][0, 0].cpu().numpy()
                    batch_gt.append(gt)
                
                # Apply regularizers
                reg_outputs = regularizer.apply_batch(batch_masks)
                
                # Store regularized outputs
                for i in range(len(batch_masks)):
                    results["rt"].append(reg_outputs["rt"][i])
                    results["rr"].append(reg_outputs["rr"][i])
                    results["fer"].append(reg_outputs["fer"][i])
                
                # Apply fusion
                fused_masks, _ = fusion.process_batch(reg_outputs, batch_gt, training=False)
                
                # Store fusion results
                for i in range(len(batch_masks)):
                    results["fusion"].append(fused_masks[i])
        
        # Save results for this state
        state_output_dir = os.path.join(output_dir, state)
        os.makedirs(state_output_dir, exist_ok=True)
        
        import numpy as np
        np.save(os.path.join(state_output_dir, "results.npy"), results)
        
        print(f"Processed {len(results['original'])} samples for {state}")
        print(f"Results saved to {state_output_dir}")
        
        # Generate sample visualizations
        visualize_results(results, state_output_dir, num_samples=min(5, len(results["original"])))
        
    print(f"\nAll states processed. Results saved to {output_dir}")


def visualize_results(results, output_dir, num_samples=5):
    """Generate sample visualizations of processing results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Select random samples
        indices = np.random.choice(len(results["original"]), 
                                  min(num_samples, len(results["original"])), 
                                  replace=False)
        
        for idx in indices:
            plt.figure(figsize=(15, 10))
            
            # Original image
            plt.subplot(2, 3, 1)
            img = results["original"][idx]
            if img.shape[0] == 3:  # Handle channel first
                img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis('off')
            
            # Mask R-CNN prediction
            plt.subplot(2, 3, 2)
            plt.imshow(results["mask_rcnn"][idx], cmap='gray')
            plt.title("Mask R-CNN")
            plt.axis('off')
            
            # RT regularization
            plt.subplot(2, 3, 3)
            plt.imshow(results["rt"][idx], cmap='gray')
            plt.title("RT Regularization")
            plt.axis('off')
            
            # RR regularization
            plt.subplot(2, 3, 4)
            plt.imshow(results["rr"][idx], cmap='gray')
            plt.title("RR Regularization")
            plt.axis('off')
            
            # FER regularization
            plt.subplot(2, 3, 5)
            plt.imshow(results["fer"][idx], cmap='gray')
            plt.title("FER Regularization")
            plt.axis('off')
            
            # Fusion result
            plt.subplot(2, 3, 6)
            plt.imshow(results["fusion"][idx], cmap='gray')
            plt.title("RL Fusion")
            plt.axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"sample_{idx}.png"), dpi=150)
            plt.close()
            
        print(f"Saved {num_samples} sample visualizations to {viz_dir}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    
    # Print GPU information
    print_gpu_info()
    
    # Execute requested mode
    if args.mode == "train":
        if not args.states:
            print("Error: Must specify states for training")
        else:
            train_pipeline(args)
    elif args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    elif args.mode == "process":
        if not args.states:
            print("Error: Must specify states for processing")
        else:
            process_states(args)
    else:
        print(f"Unknown mode: {args.mode}")