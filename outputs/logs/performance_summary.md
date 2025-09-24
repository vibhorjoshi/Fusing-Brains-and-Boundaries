# Building Footprint Extraction Performance Summary

## Overall Performance

| Method | Average IoU |
|--------|------------|
| Mask R-CNN | 0.6590 |
| RT | 0.5770 |
| RR | 0.5228 |
| FER | 0.4922 |
| RL Fusion CPU | 0.6784 |
| RL Fusion GPU | 0.7122 |

**GPU vs CPU Improvement:** 4.98% average increase in IoU

## State-by-State Analysis

| State | Best Method | Best IoU | GPU Improvement |
|-------|-------------|----------|----------------|
| Alabama | RL Fusion GPU | 0.7120 | 3.94% |
| Arizona | RL Fusion GPU | 0.7080 | 5.36% |
| Arkansas | RL Fusion GPU | 0.7240 | 4.78% |
| California | RL Fusion GPU | 0.7150 | 5.30% |
| Florida | RL Fusion GPU | 0.7020 | 5.56% |
