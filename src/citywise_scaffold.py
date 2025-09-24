from __future__ import annotations

"""
City-wise scaffold for few-shot training on USA state patches and live testing on a
Google Static Maps image (with safe fallbacks). Produces overlays and CSV outputs
under outputs/citywise_live/.

Key pieces:
- GoogleStaticMapClient: fetch a satellite tile via Google Static Maps API.
- FewShotRLPipeline: train RL fusion on a few patches from small set of states.
- Inference helpers: baseline -> regularizer -> RL fusion -> optional LapNet refine.

Usage (programmatic):
    from src.config import Config
    from src.citywise_scaffold import run_citywise_live_demo
    run_citywise_live_demo(Config())

Notes:
- Requires env var GOOGLE_MAPS_STATIC_API_KEY for real satellite imagery.
- If the key is missing or network is unavailable, falls back to using a sampled
  local state patch as the "test image" so the demo still completes.
"""

import io
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import requests
import cv2

from .config import Config
from .data_handler import RasterDataLoader
from .regularizer import HybridRegularizer
from .adaptive_fusion import AdaptiveFusion
from .evaluator import Evaluator

try:
    from .lapnet_refiner import LapNetRefiner
    _HAS_LAPNET = True
except Exception:
    _HAS_LAPNET = False


@dataclass
class City:
    name: str
    lat: float
    lon: float
    zoom: int = 18
    size: Tuple[int, int] = (640, 640)  # width, height
    maptype: str = "satellite"


class GoogleStaticMapClient:
    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 15.0):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_STATIC_API_KEY", "")
        self.timeout_s = timeout_s

    def fetch(self, city: City) -> Optional[np.ndarray]:
        """Return BGR image (np.uint8 HxWx3) or None if unavailable.
        Uses 2x scale to increase effective resolution when available.
        """
        if not self.api_key:
            return None
        base = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{city.lat},{city.lon}",
            "zoom": str(city.zoom),
            "size": f"{city.size[0]}x{city.size[1]}",
            "maptype": city.maptype,
            "format": "png",
            "scale": "2",
            "key": self.api_key,
        }
        try:
            r = requests.get(base, params=params, timeout=self.timeout_s)
            if r.status_code != 200:
                return None
            data = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None


def _ensure_dirs(cfg: Config) -> Path:
    out_dir = cfg.OUTPUT_DIR / "citywise_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    (cfg.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    return out_dir


def _to_rgb_from_gray(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    rgb = (np.stack([g, g, g], axis=-1) * 255.0).astype(np.uint8)
    return rgb


def _overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be HxWx3")
    m = (mask > 0.5).astype(np.uint8)
    overlay = img_bgr.copy()
    colored = np.zeros_like(img_bgr)
    colored[m.astype(bool)] = color
    cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


class FewShotRLPipeline:
    """Minimal few-shot trainer for RL fusion, using existing components.

    Steps per patch:
    - Make a baseline mask from channel 0 of the patch (simple threshold).
    - Apply HybridRegularizer to get variants {rt, rr, fer}.
    - Train AdaptiveFusion on reward from IoU vs GT for a few iterations.
    - For inference on a new image, baseline -> regularizer -> select_action -> fuse.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reg = HybridRegularizer(cfg)
        self.af = AdaptiveFusion(cfg)
        self.ev = Evaluator(cfg)

    @staticmethod
    def _baseline_from_patch(p3: np.ndarray) -> np.ndarray:
        a = p3[0] if p3.ndim == 3 else p3
        thr = float(a.mean() + 0.5 * a.std())
        return (a > thr).astype(np.float32)

    def train(self, patches: List[np.ndarray], gts: List[np.ndarray], rl_iters: int = 50) -> Dict:
        # Prepare reg_results pool
        reg_results = []
        for p3, gt in zip(patches, gts):
            bp = self._baseline_from_patch(p3)
            reg = self.reg.apply(bp)
            reg_results.append({"original": bp, "regularized": reg, "ground_truth": gt.astype(np.float32)})

        rewards_log: List[float] = []
        for it in range(rl_iters):
            ep = []
            for rr in reg_results[: min(12, len(reg_results))]:
                gt = rr["ground_truth"]
                state = self.af.extract_features(rr["regularized"])  # 12-dim features
                action = self.af.select_action(state, training=True)
                fused = self.af.fuse_masks(rr["regularized"], action)
                reward = self.af.compute_reward(fused, gt)
                next_state = self.af.extract_features({"rt": fused, "rr": fused, "fer": fused})
                done = reward > 0.85
                self.af.memory.push(state, action, reward, next_state, done)
                ep.append(reward)
            if len(self.af.memory) > self.cfg.RL_BATCH_SIZE:
                self.af.train_step()
            if it % 10 == 0:
                self.af.update_target_network()
            self.af.decay_epsilon()
            if ep:
                rewards_log.append(float(np.mean(ep)))

        # Return a summary of training
        return {"n_samples": len(reg_results), "rewards": rewards_log}

    def infer_on_image(self, img_bgr: np.ndarray, patch_size: int = 256, use_lapnet: bool = True) -> Dict:
        # Convert to grayscale and crop center to patch_size
        H, W, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # center crop
        y0 = max(0, (H - patch_size) // 2)
        x0 = max(0, (W - patch_size) // 2)
        crop = gray[y0:y0+patch_size, x0:x0+patch_size]
        if crop.shape != (patch_size, patch_size):
            crop = cv2.resize(gray, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        p3 = np.stack([crop, crop, crop], axis=0)
        p3 = (p3 - p3.mean()) / (p3.std() + 1e-6)
        # Baseline -> reg -> RL -> LapNet
        bp = self._baseline_from_patch(p3)
        reg = self.reg.apply(bp)
        state = self.af.extract_features(reg)
        action = self.af.select_action(state, training=False)
        fused = self.af.fuse_masks(reg, action)
        out = {"patch_rgb": _to_rgb_from_gray(crop), "baseline": bp, "fused": fused}
        if use_lapnet and _HAS_LAPNET:
            lap_mask, _ = LapNetRefiner().refine_mask(fused)
            out["lapnet"] = lap_mask
        return out


def run_citywise_live_demo(cfg: Config, city: Optional[City] = None, rl_iters: int = 50) -> Dict:
    out_dir = _ensure_dirs(cfg)
    rng = random.Random(42)
    # Default random city list (medium/small US cities for variety)
    cities = [
        City("Boise, ID", 43.6150, -116.2023, 18),
        City("Cheyenne, WY", 41.139981, -104.820246, 18),
        City("Fargo, ND", 46.8772, -96.7898, 18),
        City("Madison, WI", 43.0731, -89.4012, 18),
        City("Tallahassee, FL", 30.4383, -84.2807, 18),
        City("Albany, NY", 42.6526, -73.7562, 18),
    ]
    if city is None:
        city = rng.choice(cities)

    # Prepare few-shot training data from configured states
    loader = RasterDataLoader(cfg)
    train_states = cfg.TRAINING_STATES[:3] if cfg.TRAINING_STATES else loader.list_available_states()[:3]
    patches: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for st in train_states:
        data = loader.sample_patches_from_state(st, max_patches=min(8, cfg.MAX_PATCHES_PER_STATE))
        if data is None:
            continue
        p3, m, _ = data
        patches.extend(p3)
        masks.extend(m)
        if len(patches) >= 24:
            break
    if not patches:
        # last resort: create synthetic single patch for training
        H = W = cfg.PATCH_SIZE
        synth = np.zeros((H, W), dtype=np.float32)
        cv2.rectangle(synth, (60, 60), (180, 180), 1, -1)
        patches = [np.stack([synth, synth, synth], 0)]
        masks = [(synth > 0.5).astype(np.uint8)]

    # Train RL few-shot
    fs = FewShotRLPipeline(cfg)
    train_summary = fs.train(patches, masks, rl_iters=rl_iters)

    # Fetch live image (or fallback to a local patch rendered to RGB)
    client = GoogleStaticMapClient()
    live_img = client.fetch(city)
    source = "google_static_maps"
    if live_img is None:
        # Fallback: use the first training patch as a pseudo-image
        source = "fallback_local_patch"
        p3 = patches[0]
        gray = p3[0]
        live_img = cv2.cvtColor(_to_rgb_from_gray(gray), cv2.COLOR_RGB2BGR)

    # Inference on live image
    inf = fs.infer_on_image(live_img, patch_size=cfg.PATCH_SIZE, use_lapnet=True)
    patch_rgb = inf["patch_rgb"]  # RGB
    baseline = inf["baseline"]
    fused = inf["fused"]
    lapnet = inf.get("lapnet")

    # Save artifacts
    city_slug = city.name.replace(",", "").replace(" ", "_")
    input_path = out_dir / f"{city_slug}_input.png"
    cv2.imwrite(str(input_path), live_img)

    # Compose overlays on the RGB patch for consistency
    bgr_patch = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    ov_base = _overlay_mask(bgr_patch, baseline, (0, 165, 255))  # orange
    ov_fused = _overlay_mask(bgr_patch, fused, (40, 200, 40))    # green
    cv2.imwrite(str(out_dir / f"{city_slug}_overlay_baseline.png"), ov_base)
    cv2.imwrite(str(out_dir / f"{city_slug}_overlay_fused.png"), ov_fused)
    if lapnet is not None:
        ov_lap = _overlay_mask(bgr_patch, lapnet, (255, 0, 0))   # blue
        cv2.imwrite(str(out_dir / f"{city_slug}_overlay_lapnet.png"), ov_lap)

    # Write a small CSV summary
    import pandas as pd
    rows = [{
        "city": city.name,
        "source": source,
        "train_samples": train_summary.get("n_samples", 0),
        "rewards_len": len(train_summary.get("rewards", [])),
        "rewards_last": float(train_summary.get("rewards", [0.0])[-1]) if train_summary.get("rewards") else 0.0,
        "has_lapnet": bool(lapnet is not None),
    }]
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{city_slug}_summary.csv"
    df.to_csv(csv_path, index=False)

    return {
        "city": city.name,
        "input": str(input_path),
        "overlay_baseline": str(out_dir / f"{city_slug}_overlay_baseline.png"),
        "overlay_fused": str(out_dir / f"{city_slug}_overlay_fused.png"),
        "overlay_lapnet": str(out_dir / f"{city_slug}_overlay_lapnet.png") if lapnet is not None else None,
        "summary_csv": str(csv_path),
    }
