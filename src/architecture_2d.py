import os
import matplotlib
# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def save_2d_architecture_diagram(output_path: str = "outputs/figures/architecture_diagram.png") -> str:
    """
    Render and save a 2D Hybrid GeoAI Architecture diagram.
    
    Returns the absolute path to the saved image.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Layer 1: Raster Input Processing
    ax.add_patch(Rectangle((1, 8.5), 8, 1, fill=True, facecolor='lightblue', edgecolor='black'))
    ax.text(5, 9, 'Layer 1: Raster Input Processing', ha='center', va='center', fontsize=12)

    # Layer 2: Instance Segmentation Network
    ax.add_patch(Rectangle((1, 6.5), 8, 1, fill=True, facecolor='lightgreen', edgecolor='black'))
    ax.text(5, 7, 'Layer 2: Instance Segmentation Network (Mask R-CNN)', ha='center', va='center', fontsize=12)

    # Layer 3: Geometric Regularization Modules
    ax.add_patch(Rectangle((1, 4.5), 8, 1, fill=True, facecolor='lightyellow', edgecolor='black'))
    ax.text(5, 5, 'Layer 3: Geometric Regularization Modules (RT, RR, FER)', ha='center', va='center', fontsize=12)

    # Layer 4: Adaptive Control System
    ax.add_patch(Rectangle((1, 2.5), 8, 1, fill=True, facecolor='lightpink', edgecolor='black'))
    ax.text(5, 3, 'Layer 4: Adaptive Control System (RL DQN Fusion)', ha='center', va='center', fontsize=12)

    # Arrows connecting layers
    ax.add_patch(FancyArrowPatch((5, 8.5), (5, 7.5), arrowstyle='->', mutation_scale=20, linewidth=2))
    ax.add_patch(FancyArrowPatch((5, 6.5), (5, 5.5), arrowstyle='->', mutation_scale=20, linewidth=2))
    ax.add_patch(FancyArrowPatch((5, 4.5), (5, 3.5), arrowstyle='->', mutation_scale=20, linewidth=2))

    # Output
    ax.add_patch(Rectangle((3.5, 0.5), 3, 1, fill=True, facecolor='lavender', edgecolor='black'))
    ax.text(5, 1, 'Output: Regularized Polygons', ha='center', va='center', fontsize=12)
    ax.add_patch(FancyArrowPatch((5, 2.5), (5, 1.5), arrowstyle='->', mutation_scale=20, linewidth=2))

    plt.title('Hybrid GeoAI Architecture for Building Footprint Regularization', fontsize=14)
    
    # Add step labels beside arrows
    ax.text(5.5, 8, 'Step 1-2: Data Preparation & Training', ha='left', va='center', fontsize=10, style='italic')
    ax.text(5.5, 6, 'Step 3: Mask R-CNN Inference', ha='left', va='center', fontsize=10, style='italic')
    ax.text(5.5, 4, 'Step 4: Hybrid Regularization', ha='left', va='center', fontsize=10, style='italic')
    ax.text(5.5, 2, 'Step 5-7: RL Fusion, Post-Processing & Evaluation', ha='left', va='center', fontsize=10, style='italic')
    
    # Add some detail boxes on the right side
    detail_x = 7.5
    ax.text(detail_x, 8.8, '• GDAL Window Sampling', ha='left', va='center', fontsize=9)
    ax.text(detail_x, 8.6, '• 3-Channel Normalization', ha='left', va='center', fontsize=9)
    
    ax.text(detail_x, 6.8, '• ResNet-50 FPN Backbone', ha='left', va='center', fontsize=9)
    ax.text(detail_x, 6.6, '• Binary Instance Masks', ha='left', va='center', fontsize=9)
    
    ax.text(detail_x, 4.8, '• RT: Morphological Closing', ha='left', va='center', fontsize=9)
    ax.text(detail_x, 4.6, '• RR: Open-Close Operations', ha='left', va='center', fontsize=9)
    ax.text(detail_x, 4.4, '• FER: Edge-Aware Refinement', ha='left', va='center', fontsize=9)
    
    ax.text(detail_x, 2.8, '• DQN: 12D State, 27 Actions', ha='left', va='center', fontsize=9)
    ax.text(detail_x, 2.6, '• Reward: 0.6 IoU + 0.3 F1 + 0.1 Prec', ha='left', va='center', fontsize=9)
    ax.text(detail_x, 2.4, '• Epsilon-Greedy (Decay 0.995)', ha='left', va='center', fontsize=9)
    
    ax.text(detail_x, 0.9, '• IoU, F1, Boundary IoU, Hausdorff', ha='left', va='center', fontsize=9)
    ax.text(detail_x, 0.7, '• Shape Quality & Robustness', ha='left', va='center', fontsize=9)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return os.path.abspath(output_path)


if __name__ == "__main__":
    path = save_2d_architecture_diagram()
    print(f"2D architecture diagram saved as {path}")