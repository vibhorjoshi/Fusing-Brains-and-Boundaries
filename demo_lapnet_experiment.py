import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.lapnet_refiner import LapNetRefiner


def synthetic_rect_with_noise(h=256, w=256, rect=(60, 60, 180, 180), jitter=5):
    m = np.zeros((h, w), dtype=np.float32)
    x1, y1, x2, y2 = rect
    m[y1:y2, x1:x2] = 1.0
    # add boundary noise
    rng = np.random.default_rng(42)
    for _ in range(600):
        y = rng.integers(low=y1 - jitter, high=y2 + jitter)
        x = rng.integers(low=x1 - jitter, high=x2 + jitter)
        if 0 <= y < h and 0 <= x < w:
            m[y, x] = 1.0 if rng.random() > 0.5 else 0.0
    return m


def main():
    out_dir = Path('outputs/lapnet_demo')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic baseline and GT
    gt = synthetic_rect_with_noise(jitter=0)
    baseline = synthetic_rect_with_noise(jitter=6)  # noisier baseline mask

    refiner = LapNetRefiner(smooth_lambda=0.55, corner_preserve=0.85, angle_thresh_deg=32.0, iters=20)
    refined_mask, refined_vertices = refiner.refine_mask(baseline)

    iou_before = refiner.iou(baseline, gt)
    iou_after = refiner.iou(refined_mask, gt)

    # Save before/after 2D overlays
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(gt, cmap='gray'); axs[0].set_title('Ground Truth'); axs[0].axis('off')
    axs[1].imshow(baseline, cmap='gray'); axs[1].set_title(f'Baseline (IoU={iou_before:.3f})'); axs[1].axis('off')
    axs[2].imshow(refined_mask, cmap='gray'); axs[2].set_title(f'LapNet (IoU={iou_after:.3f})'); axs[2].axis('off')
    plt.tight_layout()
    fig.savefig(out_dir / 'lapnet_before_after_2d.png', dpi=250)
    plt.close(fig)

    # Save 3D view
    LapNetRefiner.render_3d_comparison(baseline, refined_mask, str(out_dir / 'lapnet_3d_extrusion.png'))

    # Metrics CSV
    with open(out_dir / 'metrics.csv', 'w') as f:
        f.write('metric,value\n')
        f.write(f'iou_before,{iou_before:.6f}\n')
        f.write(f'iou_after,{iou_after:.6f}\n')
        f.write(f'iou_delta,{(iou_after - iou_before):.6f}\n')

    print('LapNet demo complete:', out_dir)


if __name__ == '__main__':
    main()
