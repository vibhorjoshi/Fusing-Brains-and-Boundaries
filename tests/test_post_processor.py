import numpy as np

from src.config import Config
from src.post_processor import PostProcessor


def test_polygonization_simple_rect():
    cfg = Config()
    pp = PostProcessor(cfg)
    m = np.zeros((64, 64), dtype=np.float32)
    m[10:30, 15:40] = 1.0
    polys = pp.mask_to_polygons(m)
    assert len(polys) >= 1


def test_merge_overlapping():
    cfg = Config()
    pp = PostProcessor(cfg)
    m = np.zeros((64, 64), dtype=np.float32)
    m[10:30, 15:35] = 1.0
    m[20:40, 25:45] = 1.0
    polys = pp.mask_to_polygons(m)
    merged = pp.merge_overlapping_polygons(polys, overlap_threshold=0.05)
    assert len(merged) <= len(polys)
