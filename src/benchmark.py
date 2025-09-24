from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import cv2
import pandas as pd


def iou_prec_recall(pred: np.ndarray, gt: np.ndarray):
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    iou = inter / (union + 1e-8) if union > 0 else 0.0
    precision = inter / (pred_b.sum() + 1e-8)
    recall = inter / (gt_b.sum() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(iou), float(precision), float(recall), float(f1)


class PerformanceBenchmark:
    def __init__(self, config):
        self.config = config

    def simple_threshold_baseline(self, patch: np.ndarray) -> np.ndarray:
        # use channel 0
        area = patch[0] if patch.ndim == 3 else patch
        thr = area.mean() + 2 * area.std()
        return (area > thr).astype(np.float32)

    def morphology_baseline(self, patch: np.ndarray) -> np.ndarray:
        area = patch[0] if patch.ndim == 3 else patch
        binary = (area > area.mean()).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed.astype(np.float32)

    def watershed_baseline(self, patch: np.ndarray) -> np.ndarray:
        area = patch[0] if patch.ndim == 3 else patch
        image = (area * 255).astype(np.uint8)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        # simple proxy: threshold distance
        return (dist > (0.3 * dist.max())).astype(np.float32)

    def evaluate_baseline(self, predicted_masks: List[np.ndarray], ground_truth_masks: List[np.ndarray]) -> Dict[str, float]:
        ious, ps, rs = [], [], []
        for pred, gt in zip(predicted_masks, ground_truth_masks):
            i, p, r, _ = iou_prec_recall(pred, gt)
            ious.append(i); ps.append(p); rs.append(r)
        f1 = 2 * (np.mean(ps) * np.mean(rs)) / (np.mean(ps) + np.mean(rs) + 1e-8) if ps and rs else 0.0
        return {
            "iou": float(np.mean(ious) if ious else 0.0),
            "precision": float(np.mean(ps) if ps else 0.0),
            "recall": float(np.mean(rs) if rs else 0.0),
            "f1_score": float(f1),
        }


class ExperimentRunner:
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}

    def run_experiment(self, name: str, config, state_name: str | None = None, run_callable=None):
        if run_callable is None:
            raise ValueError("run_callable must be provided")
        results, metrics = run_callable()
        self.experiments[name] = {
            "config": config,
            "results": results,
            "metrics": metrics,
            "status": "completed",
        }

    def compare_experiments(self):
        if not self.experiments:
            return None
        rows = []
        for name, exp in self.experiments.items():
            m = exp.get("metrics", {})
            rows.append({
                "Experiment": name,
                "IoU": m.get("iou", 0),
                "Precision": m.get("precision", 0),
                "Recall": m.get("recall", 0),
                "F1-Score": m.get("f1_score", 0),
            })
        df = pd.DataFrame(rows)
        df.to_csv("./experiment_comparison.csv", index=False)
        return df
