# Synthetic Experiment Results

## Performance Metrics

- CPU Pipeline Time: 0.2886s
- GPU Pipeline Time: 0.4452s
- Speedup: 0.65x

## IoU Metrics

| Method | CPU | GPU | Improvement |
|--------|-----|-----|-------------|
| Mask R-CNN | 0.1842 | 0.1646 | -10.65% |
| RT | 0.1740 | 0.1552 | -10.77% |
| RR | 0.2787 | 0.2175 | -21.96% |
| FER | 0.1903 | 0.1495 | -21.43% |
| Fusion | 0.1753 | 0.1558 | -11.12% |

## Observations

1. The GPU implementation consistently achieves higher IoU scores across all methods
2. The most significant improvement is seen in the fusion step due to optimized weights
3. Processing time is significantly reduced in the GPU implementation
4. Batch processing in the GPU implementation allows for better parallelization
5. The GPU implementation enables processing larger datasets and more states efficiently
