import numpy as np
import cv2
import torch
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)


def _find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.max() <= 1 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt.squeeze(axis=1)
    if cnt.ndim != 2 or cnt.shape[0] < 4:
        return None
    return cnt.astype(np.float32)


def _rasterize_polygon(vertices: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    poly = vertices.reshape((-1, 1, 2)).astype(np.int32)
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [np.array(poly, dtype=np.int32)], 1)
    return canvas.astype(np.float32)


def _corner_mask(vertices: np.ndarray, angle_thresh_deg: float = 35.0) -> np.ndarray:
    n = len(vertices)
    if n < 5:
        return np.ones(n, dtype=bool)
    corners = np.zeros(n, dtype=bool)
    for i in range(n):
        p_prev = vertices[(i - 1) % n]
        p = vertices[i]
        p_next = vertices[(i + 1) % n]
        v1 = p_prev - p
        v2 = p_next - p
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cosang))
        # Mark as corner if sharper than threshold
        if angle < angle_thresh_deg or angle > (180 - angle_thresh_deg):
            corners[i] = True
    return corners


class LapNetRefiner:
    """
    Lightweight geometric refiner ("LapNet") applying constrained Laplacian smoothing
    to polygonal building footprints. Corners are preserved; non-corner vertices are
    smoothed using a learned/static per-vertex weight blending towards the Laplacian.

    This is not a heavy neural network; it's a fast, differentiable-style operator that
    mimics laplacian-based refinement practical for post-processing.
    """

    def __init__(self, smooth_lambda: float = 0.5, corner_preserve: float = 1.0,
                 angle_thresh_deg: float = 35.0, iters: int = 10):
        self.smooth_lambda = float(np.clip(smooth_lambda, 0.0, 1.0))
        self.corner_preserve = float(np.clip(corner_preserve, 0.0, 1.0))
        self.angle_thresh_deg = angle_thresh_deg
        self.iters = iters

    def refine_polygon(self, vertices: np.ndarray) -> np.ndarray:
        if vertices is None or len(vertices) < 4:
            return vertices
        V = vertices.astype(np.float32).copy()
        n = len(V)
        corner_mask = _corner_mask(V, self.angle_thresh_deg)

        for _ in range(self.iters):
            V_prev = V.copy()
            # Circular neighbors
            V_left = np.roll(V_prev, 1, axis=0)
            V_right = np.roll(V_prev, -1, axis=0)
            lap = 0.5 * (V_left + V_right) - V_prev
            V_smooth = V_prev + self.smooth_lambda * lap
            # Preserve corners more (less movement)
            w = np.where(corner_mask[:, None], (1.0 - self.corner_preserve), 1.0)
            V = V_prev * (1.0 - w) + V_smooth * w
        return V

    def refine_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Refine a binary mask by extracting its main contour, smoothing vertices,
        and rasterizing back. Returns (refined_mask, refined_vertices)."""
        cnt = _find_largest_contour(mask)
        if cnt is None:
            return mask.astype(np.float32), None
        refined = self.refine_polygon(cnt)
        out = _rasterize_polygon(refined, mask.shape)
        return out, refined

    @staticmethod
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        a1 = (a > 0.5).astype(np.uint8)
        b1 = (b > 0.5).astype(np.uint8)
        inter = np.logical_and(a1, b1).sum()
        union = np.logical_or(a1, b1).sum() + 1e-6
        return float(inter / union)

    @staticmethod
    def render_3d_comparison(before_mask: np.ndarray, after_mask: np.ndarray,
                              out_path: str, height_before: float = 10.0,
                              height_after: float = 10.0):
        """Render a simple 3D extrusion of before/after masks for visual comparison."""
        def mask_to_vertices(m):
            c = _find_largest_contour(m)
            return c if c is not None else np.array([])

        vb = mask_to_vertices(before_mask)
        va = mask_to_vertices(after_mask)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        def plot_extrusion(verts: np.ndarray, h: float, color: str, label: str):
            if verts.size == 0:
                return
            # Close polygon
            vv = np.vstack([verts, verts[0]])
            xs, ys = vv[:, 0], vv[:, 1]
            zs0 = np.zeros_like(xs)
            zs1 = np.full_like(xs, h)
            ax.plot(xs, ys, zs0, color=color, alpha=0.9, label=label)
            ax.plot(xs, ys, zs1, color=color, alpha=0.4)
            # vertical edges sampled sparsely
            for k in range(0, len(xs), max(1, len(xs)//12)):
                ax.plot([xs[k], xs[k]], [ys[k], ys[k]], [0, h], color=color, alpha=0.2)

        plot_extrusion(vb, height_before, '#d62728', 'Before (baseline)')
        plot_extrusion(va, height_after, '#2ca02c', 'After (LapNet)')

        ax.set_title('3D Extrusion: Before vs After LapNet Refinement')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Height')
        ax.view_init(elev=25, azim=-60)
        ax.legend(loc='best')
        plt.tight_layout()
        fig.savefig(out_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
