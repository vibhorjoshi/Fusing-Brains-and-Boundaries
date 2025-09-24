from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .data_handler import RasterDataLoader


def _downsample(arr: np.ndarray, target: int = 64) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    fy = max(1, h // target)
    fx = max(1, w // target)
    if arr.ndim == 3:
        return arr[:, ::fy, ::fx]
    return arr[::fy, ::fx]


def save_input_mask_3d_proofs(config, n_states: int = 2, patches_per_state: int = 2) -> List[Path]:
    """Generate figures that prove how rasters are converted to 3-channel inputs paired with binary masks,
    visualized in 3D surface style.

    For each sampled patch: plots three 3D surfaces (one per channel of the input) and one 3D surface for the mask.
    Saves figures under outputs/figures/proof_input_mask3d_<state>_<idx>.png
    Returns list of saved paths.
    """
    out_paths: List[Path] = []
    out_dir = config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = RasterDataLoader(config)
    orig_cap = config.MAX_PATCHES_PER_STATE
    config.MAX_PATCHES_PER_STATE = patches_per_state
    states = loader.load_multiple_states(limit=n_states)
    config.MAX_PATCHES_PER_STATE = orig_cap

    if not states:
        print("[proof3d] No states found; cannot generate proof figures.")
        return out_paths

    for state, (patches, masks, profile) in states.items():
        for idx, (p3, m) in enumerate(zip(patches, masks)):
            # p3: CxHxW, m: HxW
            C, H, W = p3.shape
            # downsample for plotting speed
            p3_ds = _downsample(p3, target=64)
            m_ds = _downsample(m, target=64)
            h, w = m_ds.shape
            X, Y = np.meshgrid(np.arange(w), np.arange(h))

            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f"3-Channel Input and Paired Mask (State={state}, Patch={idx})")

            # Channel 1
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax1.plot_surface(X, Y, p3_ds[0], cmap='viridis', linewidth=0, antialiased=True)
            ax1.set_title('Input Channel 1 (Z=intensity)')
            ax1.set_xlabel('X'); ax1.set_ylabel('Y')

            # Channel 2
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            z2 = p3_ds[1] if C > 1 else p3_ds[0]
            ax2.plot_surface(X, Y, z2, cmap='plasma', linewidth=0, antialiased=True)
            ax2.set_title('Input Channel 2 (Z=intensity)')
            ax2.set_xlabel('X'); ax2.set_ylabel('Y')

            # Channel 3
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            z3 = p3_ds[2] if C > 2 else p3_ds[0]
            ax3.plot_surface(X, Y, z3, cmap='cividis', linewidth=0, antialiased=True)
            ax3.set_title('Input Channel 3 (Z=intensity)')
            ax3.set_xlabel('X'); ax3.set_ylabel('Y')

            # Binary mask surface
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            ax4.plot_surface(X, Y, m_ds.astype(float), cmap='Greys', linewidth=0, antialiased=True)
            ax4.set_title('Paired Binary Mask (Z∈{0,1})')
            ax4.set_xlabel('X'); ax4.set_ylabel('Y')

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = out_dir / f"proof_input_mask3d_{state}_{idx}.png"
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            out_paths.append(save_path)

    return out_paths
