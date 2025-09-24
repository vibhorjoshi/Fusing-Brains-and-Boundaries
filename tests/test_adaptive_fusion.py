import numpy as np

from src.config import Config
from src.adaptive_fusion import AdaptiveFusion


def test_feature_vector_size():
    cfg = Config()
    af = AdaptiveFusion(cfg)
    ones = np.ones((64, 64), dtype=np.float32)
    zero = np.zeros_like(ones)
    feats = af.extract_features({"rt": ones, "rr": zero, "fer": ones})
    assert feats.shape == (12,)


def test_reward_monotonicity():
    cfg = Config()
    af = AdaptiveFusion(cfg)
    gt = np.zeros((64, 64), dtype=np.float32)
    gt[16:32, 16:32] = 1.0
    perfect = gt.copy()
    empty = np.zeros_like(gt)
    r_perfect = af.compute_reward(perfect, gt)
    r_empty = af.compute_reward(empty, gt)
    assert r_perfect > r_empty
