from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset


@dataclass
class PatchSample:
	image: np.ndarray  # CxHxW float32
	mask: np.ndarray   # HxW uint8


class RasterDataLoader:
	"""Load state rasters and derive patches and masks.
	Assumes presence of per-state rasters in `building_footprint_results/data/<StateName>/`.
	"""

	def __init__(self, config):
		self.config = config

	def _find_state_dir(self, state_name: Optional[str]) -> Optional[Path]:
		base = self.config.DATA_DIR
		if state_name:
			cand = base / state_name
			if cand.exists():
				return cand
		# try to auto-pick first state folder with .tif files
		if base.exists():
			for p in base.iterdir():
				if p.is_dir() and any(fp.suffix.lower() == ".tif" for fp in p.iterdir()):
					return p
		return None

	def list_available_states(self) -> List[str]:
		base = self.config.DATA_DIR
		states: List[str] = []
		if base.exists():
			for p in sorted(base.iterdir()):
				if p.is_dir() and any(fp.suffix.lower() == ".tif" for fp in p.iterdir()):
					states.append(p.name)
		return states

	def extract_patches_with_raw(self, img: np.ndarray, mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
		"""Return normalized 3-channel patches, masks, and raw single-band patches."""
		H, W = img.shape
		ps = self.config.PATCH_SIZE
		patches: List[np.ndarray] = []
		masks: List[np.ndarray] = []
		raw_patches: List[np.ndarray] = []
		stride = ps
		img3 = np.stack([img, img, img], axis=0)
		count = 0
		for y in range(0, H - ps + 1, stride):
			for x in range(0, W - ps + 1, stride):
				raw = img[y:y+ps, x:x+ps].astype(np.float32)
				p = img3[:, y:y+ps, x:x+ps]
				m = mask[y:y+ps, x:x+ps]
				patches.append((p - p.mean()) / (p.std() + 1e-6))
				masks.append(m.astype(np.uint8))
				raw_patches.append(raw)
				count += 1
				if count >= self.config.MAX_PATCHES_PER_STATE:
					return patches, masks, raw_patches
		return patches, masks, raw_patches

	def load_multiple_states(self, limit: int = 10) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray], Optional[dict]]]:
		"""Return dict of state -> (patches, masks, raster profile). Caps patches per state via config."""
		out: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], Optional[dict]]] = {}
		for s in self.list_available_states()[:limit]:
			img, m, profile = self.load_state_raster(s)
			if img is None or m is None:
				continue
			patches, masks = self.extract_patches(img, m)
			out[s] = (patches, masks, profile)
		return out

	@staticmethod
	def get_raster_bounds(profile: dict) -> Optional[Tuple[float, float, float, float]]:
		"""Return (minx, miny, maxx, maxy) in geographic coords if available."""
		try:
			from rasterio.transform import array_bounds
			transform = profile.get("transform")
			height = profile.get("height")
			width = profile.get("width")
			if transform is None or height is None or width is None:
				return None
			miny, minx, maxy, maxx = array_bounds(height, width, transform)
			return (minx, miny, maxx, maxy)
		except Exception:
			return None

	def load_state_raster(self, state_name: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
		state_dir = self._find_state_dir(state_name)
		if not state_dir:
			return None, None, None
		# heuristics: use *_avg.tif as image proxy and *_cnt.tif or *_sum.tif as mask proxy
		avg = next((p for p in state_dir.glob("*avg.tif")), None)
		cnt = next((p for p in state_dir.glob("*cnt.tif")), None)
		if not avg or not cnt:
			# fallback to any tif
			tifs = list(state_dir.glob("*.tif"))
			if len(tifs) < 2:
				return None, None, None
			avg, cnt = tifs[:2]
		with rasterio.open(avg) as src_avg:
			profile = src_avg.profile
		# Avoid reading full arrays here; use sampler for multi-state scenarios
		return None, None, profile

	def extract_patches(self, img: np.ndarray, mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
		H, W = img.shape
		ps = self.config.PATCH_SIZE
		patches: List[np.ndarray] = []
		masks: List[np.ndarray] = []
		stride = ps  # non-overlapping for speed
		# Build a 3-channel proxy (repeat bands)
		img3 = np.stack([img, img, img], axis=0)
		count = 0
		for y in range(0, H - ps + 1, stride):
			for x in range(0, W - ps + 1, stride):
				p = img3[:, y:y+ps, x:x+ps]
				m = mask[y:y+ps, x:x+ps]
				patches.append((p - p.mean()) / (p.std() + 1e-6))
				masks.append(m.astype(np.uint8))
				count += 1
				if count >= self.config.MAX_PATCHES_PER_STATE:
					return patches, masks
		return patches, masks

	def sample_patches_from_state(self, state_name: str, max_patches: int) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], dict]]:
		state_dir = self._find_state_dir(state_name)
		if not state_dir:
			return None
		avg = next((p for p in state_dir.glob("*avg.tif")), None)
		cnt = next((p for p in state_dir.glob("*cnt.tif")), None)
		if not avg or not cnt:
			tifs = list(state_dir.glob("*.tif"))
			if len(tifs) < 2:
				return None
			avg, cnt = tifs[:2]
		ps = self.config.PATCH_SIZE
		from rasterio.windows import Window
		patches: List[np.ndarray] = []
		masks: List[np.ndarray] = []
		import random
		with rasterio.open(avg) as src_avg, rasterio.open(cnt) as src_cnt:
			H, W = src_avg.height, src_avg.width
			profile = src_avg.profile
			trials = 0
			max_trials = max_patches * 8
			while len(patches) < max_patches and trials < max_trials:
				trials += 1
				x = random.randrange(0, max(1, W - ps))
				y = random.randrange(0, max(1, H - ps))
				win = Window(x, y, ps, ps)
				try:
					img_win = src_avg.read(1, window=win).astype(np.float32)
					cnt_win = src_cnt.read(1, window=win)
					if img_win.shape != (ps, ps) or cnt_win.shape != (ps, ps):
						continue
					img3 = np.stack([img_win, img_win, img_win], axis=0)
					p3 = (img3 - img3.mean()) / (img3.std() + 1e-6)
					thr = float(cnt_win.mean() + 0.5 * cnt_win.std())
					mask = (cnt_win > thr).astype(np.uint8)
					patches.append(p3.astype(np.float32))
					masks.append(mask)
				except Exception:
					continue
		return patches, masks, profile

	def load_multiple_states(self, limit: int = 10) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray], Optional[dict]]]:
		out: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], Optional[dict]]] = {}
		for s in self.list_available_states()[:limit]:
			data = self.sample_patches_from_state(s, self.config.MAX_PATCHES_PER_STATE)
			if data is None:
				continue
			p3, masks, profile = data
			out[s] = (p3, masks, profile)
		return out


class BuildingDataset(Dataset):
	def __init__(self, patches: List[np.ndarray], masks: List[np.ndarray]):
		self.patches = patches
		self.masks = masks

		assert len(self.patches) == len(self.masks)

	def __len__(self):
		return len(self.patches)

	def __getitem__(self, idx: int):
		x = self.patches[idx].astype(np.float32)
		y = (self.masks[idx] > 0).astype(np.uint8)
		# Return in Mask R-CNN compatible dict format
		# For simplicity, use the binary mask as one instance
		import torch
		masks = torch.from_numpy(y[None, ...].astype(np.uint8))
		boxes = self._mask_to_boxes(y)
		labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
		target = {"boxes": boxes, "labels": labels, "masks": masks}
		return x, target

	@staticmethod
	def _mask_to_boxes(mask: np.ndarray):
		import torch
		ys, xs = np.where(mask > 0)
		if ys.size == 0 or xs.size == 0:
			return torch.zeros((0, 4), dtype=torch.float32)
		y1, x1 = ys.min(), xs.min()
		y2, x2 = ys.max(), xs.max()
		return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


def collate_fn(batch):
	images, targets = list(zip(*batch))
	import torch
	images = [torch.from_numpy(im) for im in images]
	return images, list(targets)

