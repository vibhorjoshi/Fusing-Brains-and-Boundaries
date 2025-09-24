import os
import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection


def save_3d_architecture_diagram(output_path: str = "outputs/figures/architecture_diagram_3d.png") -> str:
    """
    Render and save a 3D Hybrid GeoAI Architecture diagram.

    Returns the absolute path to the saved image.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Bounds expanded to include negative y for the output block
    ax.set_xlim(0, 12)
    ax.set_ylim(-3, 12)
    ax.set_zlim(0, 6)
    ax.axis("off")

    # Helper for 3D arrows
    def arrow3d(ax_, start, end, color="black", linewidth=1.8, alpha=1.0):
        sx, sy, sz = start
        ex, ey, ez = end
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        # quiver draws a vector with an arrow head
        ax_.quiver(sx, sy, sz, dx, dy, dz, arrow_length_ratio=0.1, color=color, linewidth=linewidth, alpha=alpha)

    # Layer 1: Raster Input Processing
    ax.bar3d(1, 9, 0, 8, 1, 1, shade=True, color="lightblue", alpha=0.85)
    ax.text(5, 10, 0.5, "Layer 1: Raster Input Processing\n(GDAL, 512x512 Patches, [0,1] Norm)",
            ha="center", va="center", fontsize=10)

    # Layer 2: Instance Segmentation Network
    ax.bar3d(1, 6, 1, 8, 1, 1, shade=True, color="lightgreen", alpha=0.85)
    ax.text(5, 7, 1.5, "Layer 2: Instance Segmentation\n(Mask R-CNN, ResNet-50, IoU>0.5)",
            ha="center", va="center", fontsize=10)

    # Layer 3: Geometric Regularization Modules
    ax.bar3d(1, 3, 2, 8, 1, 1, shade=True, color="lightyellow", alpha=0.9)
    ax.text(5, 4, 2.5, "Layer 3: Regularization\n(RT: τ=50–150, RR: θ=90°±5°, FER: Sobel)",
            ha="center", va="center", fontsize=10)

    # Layer 4: Adaptive Control System
    ax.bar3d(1, 0, 3, 8, 1, 1, shade=True, color="lightpink", alpha=0.9)
    ax.text(5, 1, 3.5, "Layer 4: Adaptive Control\n(RL DQN, State–Action–Reward, ε=0.995 Decay)",
            ha="center", va="center", fontsize=10)

    # Output
    ax.bar3d(4, -2, 4, 4, 1, 1, shade=True, color="lavender", alpha=0.9)
    ax.text(6, -1, 4.5, "Output: Vector Polygons\n(IoU>0.9, Recall>0.85)",
            ha="center", va="center", fontsize=10)

    # Arrows with parameter annotations (true 3D)
    arrows = [
        ((5, 10, 0.5), (5, 7, 1.5), "Patch Size: 512×512, Overlap: 25%"),
        ((5, 7, 1.5), (5, 4, 2.5), "NMS: 0.5, Score: 0.7"),
        ((5, 4, 2.5), (5, 1, 3.5), "Weights: RT/RR/FER Blend"),
        ((5, 1, 3.5), (6, -1, 4.5), "Merge Overlap: >10%"),
    ]
    for (start, end, label) in arrows:
        arrow3d(ax, start, end, color="black", linewidth=2.0, alpha=0.95)
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        mz = (start[2] + end[2]) / 2
        ax.text(mx, my, mz, label, ha="center", va="bottom", fontsize=8, color="black")

    # 3D Style Justification
    ax.view_init(elev=20, azim=-60)  # Perspective angle for 3D effect
    ax.set_title("3D Hybrid GeoAI Architecture for Building Footprint Regularization", pad=20, fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


if __name__ == "__main__":
    path = save_3d_architecture_diagram()
    print(f"Enhanced 3D diagram saved as {path}")
