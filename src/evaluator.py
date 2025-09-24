from __future__ import annotations

from typing import Dict, List

# Use cloud-compatible OpenCV
try:
    import cv2
except ImportError:
    from .cv2_cloud_compat import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Evaluator:
	"""Comprehensive evaluation of building footprint extraction."""

	def __init__(self, config):
		self.config = config
		self.metrics_history: List[Dict[str, float]] = []

	def compute_metrics(self, predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Dict[str, float]:
		pred = (predicted_mask > 0.5).astype(np.uint8)
		gt = (ground_truth_mask > 0.5).astype(np.uint8)

		inter = np.logical_and(pred, gt).sum()
		union = np.logical_or(pred, gt).sum()
		iou = inter / (union + 1e-8) if union > 0 else 0.0
		precision = inter / (pred.sum() + 1e-8)
		recall = inter / (gt.sum() + 1e-8)
		f1 = 2 * precision * recall / (precision + recall + 1e-8)

		hausdorff = self.compute_hausdorff_distance(pred, gt)
		boundary_iou = self.compute_boundary_iou(pred, gt)

		return {
			"iou": float(iou),
			"precision": float(precision),
			"recall": float(recall),
			"f1_score": float(f1),
			"hausdorff_distance": float(hausdorff),
			"boundary_iou": float(boundary_iou),
		}

	def compute_hausdorff_distance(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
		pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not pred_contours or not gt_contours:
			return float("inf")
		pred_points = pred_contours[0].reshape(-1, 2)
		gt_points = gt_contours[0].reshape(-1, 2)
		d1 = self.directed_hausdorff(pred_points, gt_points)
		d2 = self.directed_hausdorff(gt_points, pred_points)
		return float(max(d1, d2))

	@staticmethod
	def directed_hausdorff(points1: np.ndarray, points2: np.ndarray) -> float:
		max_min = 0.0
		for p1 in points1:
			min_d = float("inf")
			for p2 in points2:
				d = float(np.linalg.norm(p1 - p2))
				if d < min_d:
					min_d = d
			if min_d > max_min:
				max_min = min_d
		return float(max_min)

	def compute_boundary_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
		pred_b = cv2.Canny((pred_mask * 255).astype(np.uint8), 50, 150)
		gt_b = cv2.Canny((gt_mask * 255).astype(np.uint8), 50, 150)
		kernel = np.ones((3, 3), np.uint8)
		pred_b = cv2.dilate(pred_b, kernel, iterations=1)
		gt_b = cv2.dilate(gt_b, kernel, iterations=1)
		inter = np.logical_and(pred_b > 0, gt_b > 0).sum()
		union = np.logical_or(pred_b > 0, gt_b > 0).sum()
		return float(inter / (union + 1e-8)) if union > 0 else 0.0

	def evaluate_batch(self, predicted_masks: List[np.ndarray], ground_truth_masks: List[np.ndarray]) -> Dict[str, float]:
		batch = []
		for pred, gt in zip(predicted_masks, ground_truth_masks):
			batch.append(self.compute_metrics(pred, gt))
		if not batch:
			return {}
		agg: Dict[str, float] = {}
		keys = batch[0].keys()
		for k in keys:
			vals = [m[k] for m in batch if np.isfinite(m[k])]
			agg[k] = float(np.mean(vals)) if vals else 0.0
			agg[f"{k}_std"] = float(np.std(vals)) if vals else 0.0
		self.metrics_history.append(agg)
		return agg

	def plot_metrics_evolution(self):
		if not self.metrics_history:
			return
		df = pd.DataFrame(self.metrics_history)
		fig, axes = plt.subplots(2, 3, figsize=(15, 10))
		axes = axes.flatten()
		metrics = ["iou", "precision", "recall", "f1_score", "hausdorff_distance", "boundary_iou"]
		for i, m in enumerate(metrics):
			if m in df.columns:
				axes[i].plot(df[m])
				axes[i].set_title(m.replace("_", " ").title())
				axes[i].set_xlabel("Iteration")
				axes[i].grid(True)
		fig.tight_layout()
		save_path = self.config.FIGURES_DIR / "metrics_evolution.png"
		fig.savefig(save_path, dpi=self.config.SAVE_FIGURE_DPI, bbox_inches="tight")
		plt.close(fig)

