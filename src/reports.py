from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# Helper to draw a simple table as PNG
def _save_table_png(df: pd.DataFrame, fig_path, title: str = "Demo Results Table"):
    fig, ax = plt.subplots(figsize=(min(16, 2 + 0.8 * len(df.columns)), 1.5 + 0.4 * (len(df) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=[[f"{v:.3f}" if isinstance(v, (int, float, np.floating)) else str(v) for v in row] for row in df.values],
                   colLabels=list(df.columns), loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_iou_comparison(fig_path, labels: List[str], ious: List[float]):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, ious, color=["#6baed6" if l != "RL Fusion" else "#31a354" for l in labels])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, max(1.0, max(ious) * 1.1))
    for i, v in enumerate(ious):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_title("Comparison: Traditional vs RT/RR/FER vs RL Fusion")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def draw_architecture_pipeline(fig_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f0f0f0"):
        rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    # Layout
    x0, y0, w, h, gap = 0.02, 0.35, 0.14, 0.3, 0.02
    x = x0
    box(x, y0, w, h, "Input Raster\n(State Patches)")
    x += w + gap
    box(x, y0, w, h, "Step 2:\nMask R-CNN\nTraining/Inference")
    x += w + gap
    box(x, y0, w, h, "Step 3:\nInitial Masks")
    x += w + gap
    box(x, y0, w, h, "Step 4:\nRegularization\n(RT, RR, FER)")
    x += w + gap
    box(x, y0, w, h, "Step 5:\nRL Adaptive Fusion\n(DQN)", fc="#e5f5e0")
    x += w + gap
    box(x, y0, w, h, "Step 6:\nPost-Process &\nVectorization")
    x += w + gap
    box(x, y0, w, h, "Step 7:\nEvaluation\n(IoU/F1 etc.)")

    # Arrows
    def arrow(x1, x2):
        ax.annotate("", xy=(x2, y0 + h / 2), xytext=(x1, y0 + h / 2),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    xs = [x0 + w, x0 + 2*w + gap, x0 + 3*w + 2*gap, x0 + 4*w + 3*gap,
          x0 + 5*w + 4*gap, x0 + 6*w + 5*gap]
    for i in range(len(xs)):
        arrow(xs[i], xs[i] + gap)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_demo_result_table(pipeline, n_samples: int = 12) -> Tuple[pd.DataFrame, str, str]:
    """Create a demo result table across methods with columns:
    Method, ShapeQuality, Robustness, Adaptability, LearningCapability, Accuracy

    - ShapeQuality: mean boundary IoU
    - Robustness: 1 - normalized mean Hausdorff (normalized by patch diagonal)
    - Adaptability: relative improvement over Threshold IoU (clipped [0,1])
    - LearningCapability: reward slope (RL only), others 0
    - Accuracy: mean IoU

    Saves CSV and PNG under outputs and returns (df, csv_path, png_path).
    """
    # Generate synthetic dataset
    patches, gts = pipeline.create_synthetic_data(n_samples)

    # Produces predictions per method
    def thresh(p):
        a = p[0] if p.ndim == 3 else p
        t = a.mean() + 2 * a.std()
        return (a > t).astype(np.float32)

    def morph(p):
        import cv2, numpy as np
        a = p[0] if p.ndim == 3 else p
        b = (a > a.mean()).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        o = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
        return o.astype(np.float32)

    regs = [pipeline.regularizer.apply((p[0] if p.ndim == 3 else p)) for p in patches]

    # RL fusion
    reg_results = [{
        "original": (p[0] if p.ndim == 3 else p),
        "regularized": r,
        "ground_truth": g
    } for p, r, g in zip(patches, regs, gts)]
    fused, rewards = pipeline.step5_adaptive_fusion(reg_results, training_iterations=40)

    # Build per-method predictions
    methods = {
        "Threshold": [thresh(p) for p in patches],
        "Morphology": [morph(p) for p in patches],
        "RT": [r["rt"] for r in regs],
        "RR": [r["rr"] for r in regs],
        "FER": [r["fer"] for r in regs],
        "RL Fusion": [fr["fused"] for fr in fused],
    }

    # Evaluate
    evalr = pipeline.evaluator
    stats: Dict[str, Dict[str, float]] = {}
    # precompute diagonal for robustness normalization
    H = W = pipeline.config.PATCH_SIZE
    diag = float(np.sqrt(H * H + W * W))
    # compute threshold baseline IoU for adaptability
    thr_metrics = [evalr.compute_metrics(pred, gt) for pred, gt in zip(methods["Threshold"], gts)]
    thr_iou_mean = float(np.mean([m["iou"] for m in thr_metrics])) if thr_metrics else 0.0

    for name, preds in methods.items():
        ms = [evalr.compute_metrics(pred, gt) for pred, gt in zip(preds, gts)]
        if not ms:
            stats[name] = {"accuracy": 0.0, "shape_quality": 0.0, "robustness": 0.0, "adaptability": 0.0, "learning_capability": 0.0}
            continue
        acc = float(np.mean([m["iou"] for m in ms]))
        shapeq = float(np.mean([m["boundary_iou"] for m in ms]))
        mean_hd = float(np.mean([m["hausdorff_distance"] if np.isfinite(m["hausdorff_distance"]) else diag for m in ms]))
        robust = float(np.clip(1.0 - (mean_hd / (diag + 1e-8)), 0.0, 1.0))
        adapt = float(np.clip((acc - thr_iou_mean) / (thr_iou_mean + 1e-6), 0.0, 1.0)) if thr_iou_mean > 0 else (1.0 if acc > 0 else 0.0)
        learn = 0.0
        if name == "RL Fusion" and rewards:
            k = max(3, min(10, len(rewards)//5))
            early = float(np.mean(rewards[:k])) if k > 0 else 0.0
            late = float(np.mean(rewards[-k:])) if k > 0 else 0.0
            learn = float(np.clip((late - early) / (abs(early) + 1e-6), 0.0, 1.0))
        stats[name] = {
            "accuracy": acc,
            "shape_quality": shapeq,
            "robustness": robust,
            "adaptability": adapt,
            "learning_capability": learn,
        }

    # Assemble DataFrame
    order = ["Threshold", "Morphology", "RT", "RR", "FER", "RL Fusion"]
    rows = []
    for name in order:
        s = stats.get(name, {})
        rows.append({
            "Method": name,
            "ShapeQuality": s.get("shape_quality", 0.0),
            "Robustness": s.get("robustness", 0.0),
            "Adaptability": s.get("adaptability", 0.0),
            "LearningCapability": s.get("learning_capability", 0.0),
            "Accuracy": s.get("accuracy", 0.0),
        })
    df = pd.DataFrame(rows)

    # Save CSV and PNG
    csv_path = pipeline.config.LOGS_DIR / "demo_result_table.csv"
    png_path = pipeline.config.FIGURES_DIR / "demo_result_table.png"
    df.to_csv(csv_path, index=False)
    _save_table_png(df, png_path, title="Demo Results: Shape/Robustness/Adaptability/Learning/Accuracy")
    return df, str(csv_path), str(png_path)
