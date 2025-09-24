#!/usr/bin/env python3
from __future__ import annotations

"""
Live city-wise demonstration.
Trains the RL fusion briefly on a few sampled state patches, then fetches a
random US city satellite image via Google Static Maps (if API key provided) and
applies the pipeline (baseline -> regularizer -> RL -> LapNet). Saves overlays
and a short CSV summary under outputs/citywise_live/.

Env var:
  GOOGLE_MAPS_STATIC_API_KEY  (optional; fallback works without it)
"""

from src.config import Config
from src.citywise_scaffold import run_citywise_live_demo


def main():
    cfg = Config()
    res = run_citywise_live_demo(cfg, city=None, rl_iters=40)
    print("City-wise live demo complete:")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
