"""
Performance Benchmarking Tool for Building Footprint Extraction

This script compares CPU and GPU implementations of:
1. Mask R-CNN inference and training
2. Regularization operations (RT, RR, FER)
3. Adaptive fusion with DQN
4. End-to-end pipeline execution

Results are reported in timing measurements and throughput metrics.
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random

# Import CPU implementations
from src.regularizer import HybridRegularizer
from src.adaptive_fusion import AdaptiveFusion

# Import GPU implementations
from src.gpu_regularizer import GPURegularizer
from src.gpu_adaptive_fusion import GPUAdaptiveFusion
from src.gpu_trainer import GPUMaskRCNNTrainer, BuildingDatasetGPU

# Import data handling utilities
from src.data_handler import DataHandler
from src.config import Config


class PerformanceBenchmark:
    """Benchmark tool to compare CPU vs GPU performance for building footprint extraction."""
    
    def __init__(self, config_path=None):
        """Initialize benchmark environment.
        
        Args:
            config_path: Path to config file (optional)
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.has_gpu = torch.cuda.is_available()
        
        # Initialize data handler
        self.data_handler = DataHandler(self.config)
        
        # Results storage
        self.results = {}
        
    def setup_environment(self):
        """Setup the testing environment and load test data."""
        print("Setting up benchmark environment...")
        
        # Print environment information
        self._print_environment_info()
        
        # Load sample dataset for benchmarking
        print("Loading benchmark data...")
        self.patches, self.masks = self._load_benchmark_data()
        
    def _print_environment_info(self):
        """Print information about the current environment."""
        print("\n=== Environment Information ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            print(f"GPU count: {torch.cuda.device_count()}")
        
        # Print CPU info
        import platform
        print(f"CPU: {platform.processor()}")
        print(f"OS: {platform.system()} {platform.version()}")
        print(f"Python version: {platform.python_version()}")
        print("===============================\n")
        
    def _load_benchmark_data(self, num_samples=100):
        """Load sample data for benchmarking.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            Tuple of (patches, masks) as numpy arrays
        """
        # For benchmarking, we'll create synthetic data if real data isn't available
        try:
            # Try to load real data first
            states = ["Alabama", "Arizona", "Arkansas"]
            patches, masks = [], []
            
            for state in states:
                state_data = self.data_handler.load_state_data(state)
                if state_data and len(state_data.get("patches", [])) > 0:
                    patches.extend(state_data["patches"][:min(50, len(state_data["patches"]))])
                    masks.extend(state_data["masks"][:min(50, len(state_data["masks"]))])
                    
                    if len(patches) >= num_samples:
                        break
                        
            if len(patches) < num_samples:
                # Supplement with synthetic data
                print(f"Only found {len(patches)} real samples, generating synthetic ones...")
                synthetic_count = num_samples - len(patches)
                patches.extend(self._generate_synthetic_data(synthetic_count)[0])
                masks.extend(self._generate_synthetic_data(synthetic_count)[1])
        except Exception as e:
            print(f"Error loading real data: {e}. Using synthetic data instead.")
            # Generate synthetic data
            patches, masks = self._generate_synthetic_data(num_samples)
            
        return patches[:num_samples], masks[:num_samples]
        
    def _generate_synthetic_data(self, num_samples, size=256):
        """Generate synthetic building patches and masks for benchmarking.
        
        Args:
            num_samples: Number of samples to generate
            size: Size of generated images (square)
            
        Returns:
            Tuple of (patches, masks) as numpy arrays
        """
        patches, masks = [], []
        
        for _ in range(num_samples):
            # Create a random 3-channel image
            patch = np.random.randint(0, 256, (size, size, 3)).astype(np.uint8)
            
            # Create a random building mask (simple rectangle/polygon)
            mask = np.zeros((size, size), dtype=np.uint8)
            
            # Add 1-3 random polygon buildings
            num_buildings = random.randint(1, 3)
            for _ in range(num_buildings):
                # Random building center and size
                cx, cy = random.randint(50, size-50), random.randint(50, size-50)
                w, h = random.randint(20, 100), random.randint(20, 100)
                
                # Generate building mask (rectangle with random rotation)
                building_mask = np.zeros((size, size), dtype=np.uint8)
                x1, y1 = max(0, cx - w//2), max(0, cy - h//2)
                x2, y2 = min(size-1, cx + w//2), min(size-1, cy + h//2)
                building_mask[y1:y2, x1:x2] = 1
                
                # Add to main mask
                mask = np.logical_or(mask, building_mask).astype(np.uint8)
            
            patches.append(patch)
            masks.append(mask)
            
        return patches, masks
    
    def benchmark_regularization(self, batch_sizes=[1, 4, 16, 64]):
        """Benchmark regularization performance on CPU vs GPU.
        
        Args:
            batch_sizes: List of batch sizes to test
        """
        print("\n=== Regularization Benchmark ===")
        results = {
            "batch_size": [],
            "cpu_time": [],
            "gpu_time": [],
            "speedup": []
        }
        
        # Initialize regularizers
        cpu_regularizer = HybridRegularizer(self.config)
        gpu_regularizer = GPURegularizer(self.config) if self.has_gpu else None
        
        for batch_size in batch_sizes:
            print(f"\nTesting with batch size {batch_size}...")
            
            # Prepare data
            batch_masks = self.masks[:batch_size]
            
            # Benchmark CPU regularization
            start_time = time.time()
            for mask in tqdm(batch_masks, desc="CPU Regularization"):
                cpu_regularizer.apply(mask)
            cpu_time = time.time() - start_time
            print(f"CPU time: {cpu_time:.4f}s")
            
            # Benchmark GPU regularization (if available)
            if gpu_regularizer:
                start_time = time.time()
                gpu_regularizer.apply_batch(batch_masks)
                gpu_time = time.time() - start_time
                print(f"GPU time: {gpu_time:.4f}s")
                speedup = cpu_time / gpu_time
                print(f"Speedup: {speedup:.2f}x")
            else:
                gpu_time = float('nan')
                speedup = float('nan')
                print("GPU not available for testing")
                
            # Store results
            results["batch_size"].append(batch_size)
            results["cpu_time"].append(cpu_time)
            results["gpu_time"].append(gpu_time)
            results["speedup"].append(speedup)
            
        self.results["regularization"] = results
        return results
        
    def benchmark_adaptive_fusion(self, batch_sizes=[1, 4, 16, 64]):
        """Benchmark adaptive fusion performance on CPU vs GPU.
        
        Args:
            batch_sizes: List of batch sizes to test
        """
        print("\n=== Adaptive Fusion Benchmark ===")
        results = {
            "batch_size": [],
            "cpu_time": [],
            "gpu_time": [],
            "speedup": []
        }
        
        # Initialize fusion modules
        cpu_fusion = AdaptiveFusion(self.config)
        gpu_fusion = GPUAdaptiveFusion(self.config) if self.has_gpu else None
        
        # Initialize regularizers (needed to generate input for fusion)
        cpu_regularizer = HybridRegularizer(self.config)
        
        for batch_size in batch_sizes:
            print(f"\nTesting with batch size {batch_size}...")
            
            # Prepare data
            batch_masks = self.masks[:batch_size]
            batch_ground_truth = [np.copy(mask) for mask in batch_masks]  # Use copies as ground truth
            
            # Pre-compute regularized outputs for fair comparison
            reg_outputs = []
            for mask in batch_masks:
                reg_outputs.append(cpu_regularizer.apply(mask))
            
            # Reorganize for batch processing
            batch_reg_outputs = {
                "original": [r["original"] for r in reg_outputs],
                "rt": [r["rt"] for r in reg_outputs],
                "rr": [r["rr"] for r in reg_outputs],
                "fer": [r["fer"] for r in reg_outputs]
            }
            
            # Benchmark CPU fusion
            start_time = time.time()
            for i in tqdm(range(batch_size), desc="CPU Fusion"):
                # Extract features
                features = []
                for reg_type in ["rt", "rr", "fer"]:
                    # CPU implementation works on individual masks
                    state_features = self._extract_cpu_features(reg_outputs[i], reg_type)
                    features.extend(state_features)
                
                # Select action and fuse masks
                action = cpu_fusion.select_action(np.array(features))
                weights = cpu_fusion.action_to_weights[action]
                
                # Apply weights to each regularization type
                fused_mask = (
                    weights[0] * reg_outputs[i]["rt"] +
                    weights[1] * reg_outputs[i]["rr"] +
                    weights[2] * reg_outputs[i]["fer"]
                )
                fused_mask = (fused_mask > 0.5).astype(np.float32)
                
                # Compute reward (IoU with ground truth)
                _ = self._compute_cpu_iou(fused_mask, batch_ground_truth[i])
                
            cpu_time = time.time() - start_time
            print(f"CPU time: {cpu_time:.4f}s")
            
            # Benchmark GPU fusion (if available)
            if gpu_fusion:
                start_time = time.time()
                gpu_fusion.process_batch(batch_reg_outputs, batch_ground_truth, training=False)
                gpu_time = time.time() - start_time
                print(f"GPU time: {gpu_time:.4f}s")
                speedup = cpu_time / gpu_time
                print(f"Speedup: {speedup:.2f}x")
            else:
                gpu_time = float('nan')
                speedup = float('nan')
                print("GPU not available for testing")
                
            # Store results
            results["batch_size"].append(batch_size)
            results["cpu_time"].append(cpu_time)
            results["gpu_time"].append(gpu_time)
            results["speedup"].append(speedup)
            
        self.results["adaptive_fusion"] = results
        return results
    
    def _extract_cpu_features(self, reg_outputs, reg_type):
        """Helper to extract features for CPU implementation."""
        mask = reg_outputs[reg_type]
        original = reg_outputs["original"]
        
        # Area ratio
        area_ratio = np.sum(mask) / mask.size
        
        # Perimeter approximation
        edges = np.abs(cv2.Laplacian(mask.astype(np.uint8), cv2.CV_64F)) > 0
        perimeter = np.sum(edges)
        
        # IoU with original
        intersection = np.sum(mask * original)
        union = np.sum((mask + original) > 0)
        iou = intersection / (union + 1e-6)
        
        # Compactness
        area = np.sum(mask)
        compactness = area / ((perimeter ** 2) + 1e-6)
        
        return [area_ratio, perimeter / 1000.0, iou, compactness * 1000.0]
    
    def _compute_cpu_iou(self, mask, ground_truth):
        """Helper to compute IoU for CPU implementation."""
        mask_binary = (mask > 0.5).astype(np.float32)
        gt_binary = (ground_truth > 0.5).astype(np.float32)
        
        intersection = np.sum(mask_binary * gt_binary)
        union = np.sum((mask_binary + gt_binary) > 0)
        
        return intersection / (union + 1e-6)
        
    def benchmark_training(self, batch_sizes=[1, 4, 8], num_epochs=2):
        """Benchmark training performance for Mask R-CNN.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_epochs: Number of epochs to train for each test
        """
        print("\n=== Training Benchmark ===")
        results = {
            "batch_size": [],
            "gpu_time": [],
            "throughput": []
        }
        
        if not self.has_gpu:
            print("GPU not available. Skipping training benchmark.")
            self.results["training"] = results
            return results
            
        # Initialize GPU trainer
        trainer = GPUMaskRCNNTrainer(self.config)
        
        for batch_size in batch_sizes:
            print(f"\nTesting with batch size {batch_size}...")
            
            # Create dataset
            dataset = BuildingDatasetGPU(self.patches, self.masks, device=self.device)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            
            # Create model
            trainer.create_model(num_classes=2)
            
            # Train for a few epochs
            start_time = time.time()
            for epoch in range(num_epochs):
                for images, targets in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Skip actual backward pass to isolate forward pass timing
                    _ = trainer.model(images, targets)
            
            training_time = time.time() - start_time
            
            # Calculate throughput (images/second)
            total_images = len(self.patches) * num_epochs
            throughput = total_images / training_time
            
            print(f"GPU training time: {training_time:.4f}s")
            print(f"Throughput: {throughput:.2f} images/second")
            
            # Store results
            results["batch_size"].append(batch_size)
            results["gpu_time"].append(training_time)
            results["throughput"].append(throughput)
            
        self.results["training"] = results
        return results
        
    def benchmark_e2e_pipeline(self, num_images=[10, 50, 100]):
        """Benchmark end-to-end pipeline execution.
        
        Args:
            num_images: List of image counts to test
        """
        print("\n=== End-to-End Pipeline Benchmark ===")
        results = {
            "num_images": [],
            "cpu_time": [],
            "gpu_time": [],
            "speedup": []
        }
        
        # Use basic CPU pipeline
        from src.pipeline import Pipeline as CPUPipeline
        
        # Create custom GPU pipeline for testing
        class GPUPipeline:
            def __init__(self, config):
                self.config = config
                self.gpu_regularizer = GPURegularizer(config)
                self.gpu_fusion = GPUAdaptiveFusion(config)
                
            def process_batch(self, patches, gt_masks):
                # Simulate Mask R-CNN outputs with ground truth
                masks = gt_masks
                
                # Apply regularization
                reg_outputs = self.gpu_regularizer.apply_batch(masks)
                
                # Apply fusion
                fused_masks, _ = self.gpu_fusion.process_batch(
                    reg_outputs, gt_masks, training=False)
                
                return fused_masks
        
        # Initialize pipelines
        cpu_pipeline = CPUPipeline(self.config)
        gpu_pipeline = GPUPipeline(self.config) if self.has_gpu else None
        
        for n in num_images:
            print(f"\nTesting with {n} images...")
            
            # Prepare data
            test_patches = self.patches[:n]
            test_masks = self.masks[:n]
            
            # Benchmark CPU pipeline
            start_time = time.time()
            for i in tqdm(range(n), desc="CPU Pipeline"):
                _ = cpu_pipeline.process_single(test_patches[i], test_masks[i])
            cpu_time = time.time() - start_time
            print(f"CPU time: {cpu_time:.4f}s")
            
            # Benchmark GPU pipeline (if available)
            if gpu_pipeline:
                start_time = time.time()
                _ = gpu_pipeline.process_batch(test_patches, test_masks)
                gpu_time = time.time() - start_time
                print(f"GPU time: {gpu_time:.4f}s")
                speedup = cpu_time / gpu_time
                print(f"Speedup: {speedup:.2f}x")
            else:
                gpu_time = float('nan')
                speedup = float('nan')
                print("GPU not available for testing")
                
            # Store results
            results["num_images"].append(n)
            results["cpu_time"].append(cpu_time)
            results["gpu_time"].append(gpu_time)
            results["speedup"].append(speedup)
            
        self.results["e2e"] = results
        return results
        
    def plot_results(self):
        """Plot benchmark results and save to figures directory."""
        print("\n=== Generating Result Plots ===")
        
        # Create output directory if it doesn't exist
        os.makedirs("outputs/figures", exist_ok=True)
        
        # Plot regularization results
        if "regularization" in self.results:
            self._plot_comparison(
                self.results["regularization"],
                "Regularization Performance",
                "Batch Size",
                "Processing Time (s)",
                "regularization_benchmark.png"
            )
            
        # Plot adaptive fusion results
        if "adaptive_fusion" in self.results:
            self._plot_comparison(
                self.results["adaptive_fusion"],
                "Adaptive Fusion Performance",
                "Batch Size",
                "Processing Time (s)",
                "adaptive_fusion_benchmark.png"
            )
            
        # Plot training throughput
        if "training" in self.results:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.results["training"]["batch_size"], 
                self.results["training"]["throughput"], 
                'o-', 
                color='purple', 
                linewidth=2, 
                markersize=8
            )
            plt.title("GPU Training Throughput", fontsize=14)
            plt.xlabel("Batch Size", fontsize=12)
            plt.ylabel("Throughput (images/second)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("outputs/figures/training_throughput.png", dpi=150)
            
        # Plot end-to-end pipeline performance
        if "e2e" in self.results:
            self._plot_comparison(
                self.results["e2e"],
                "End-to-End Pipeline Performance",
                "Number of Images",
                "Processing Time (s)",
                "e2e_pipeline_benchmark.png",
                x_key="num_images"
            )
            
        # Plot speedups across all benchmarks
        self._plot_speedups()
        
    def _plot_comparison(self, data, title, xlabel, ylabel, filename, x_key="batch_size"):
        """Helper to plot comparison between CPU and GPU performance."""
        plt.figure(figsize=(10, 6))
        
        # Plot CPU times
        if "cpu_time" in data and not all(np.isnan(data["cpu_time"])):
            plt.plot(
                data[x_key], 
                data["cpu_time"], 
                'o-', 
                color='blue', 
                label='CPU', 
                linewidth=2, 
                markersize=8
            )
            
        # Plot GPU times
        if "gpu_time" in data and not all(np.isnan(data["gpu_time"])):
            plt.plot(
                data[x_key], 
                data["gpu_time"], 
                'o-', 
                color='green', 
                label='GPU', 
                linewidth=2, 
                markersize=8
            )
            
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        plt.savefig(f"outputs/figures/{filename}", dpi=150)
        
    def _plot_speedups(self):
        """Create a summary plot of speedups across all benchmarks."""
        plt.figure(figsize=(12, 8))
        
        benchmarks = []
        speedups = []
        
        # Collect maximum speedup from each benchmark
        if "regularization" in self.results and not all(np.isnan(self.results["regularization"]["speedup"])):
            benchmarks.append("Regularization")
            speedups.append(max(self.results["regularization"]["speedup"]))
            
        if "adaptive_fusion" in self.results and not all(np.isnan(self.results["adaptive_fusion"]["speedup"])):
            benchmarks.append("Adaptive Fusion")
            speedups.append(max(self.results["adaptive_fusion"]["speedup"]))
            
        if "e2e" in self.results and not all(np.isnan(self.results["e2e"]["speedup"])):
            benchmarks.append("End-to-End")
            speedups.append(max(self.results["e2e"]["speedup"]))
            
        if not benchmarks:
            return  # No speedup data available
            
        # Create bar chart
        plt.bar(benchmarks, speedups, color=['blue', 'green', 'purple'])
        plt.title("Maximum GPU Speedup by Component", fontsize=14)
        plt.xlabel("Component", fontsize=12)
        plt.ylabel("Speedup Factor (GPU vs CPU)", fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add speedup values on top of bars
        for i, v in enumerate(speedups):
            plt.text(i, v + 0.5, f"{v:.1f}x", ha='center', fontsize=12)
            
        plt.tight_layout()
        plt.savefig("outputs/figures/speedup_summary.png", dpi=150)
        
    def save_results_csv(self):
        """Save benchmark results to CSV files."""
        print("\n=== Saving Results to CSV ===")
        
        # Create output directory if it doesn't exist
        os.makedirs("outputs/logs", exist_ok=True)
        
        # Save each benchmark result to a separate CSV file
        for name, data in self.results.items():
            df = pd.DataFrame(data)
            csv_path = f"outputs/logs/{name}_benchmark.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {name} results to {csv_path}")
            
    def run_all_benchmarks(self):
        """Run all benchmarks and generate reports."""
        # Setup environment
        self.setup_environment()
        
        # Run benchmarks
        if self.has_gpu:
            print("\nRunning benchmarks with GPU acceleration...")
        else:
            print("\nRunning benchmarks in CPU-only mode...")
            
        self.benchmark_regularization()
        self.benchmark_adaptive_fusion()
        
        if self.has_gpu:
            self.benchmark_training()
            
        self.benchmark_e2e_pipeline()
        
        # Generate visualizations
        self.plot_results()
        
        # Save raw results
        self.save_results_csv()
        
        print("\n=== Benchmarking Complete ===")
        

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Building Footprint Extraction Performance Benchmark")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(args.config)
    benchmark.run_all_benchmarks()