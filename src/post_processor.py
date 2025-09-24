from typing import List

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


class PostProcessor:
	"""Post-processing and vectorization of building masks."""

	def __init__(self, config):
		self.config = config

	def clean_mask(self, mask: np.ndarray) -> np.ndarray:
		kernel_small = np.ones((3, 3), np.uint8)
		cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
		kernel_large = np.ones((5, 5), np.uint8)
		cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
		return cleaned.astype(np.float32)

	def mask_to_polygons(self, mask: np.ndarray) -> List[Polygon]:
		cleaned = self.clean_mask(mask)
		contours, _ = cv2.findContours((cleaned * 255).astype(np.uint8),
									   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		polygons: List[Polygon] = []
		for contour in contours:
			if len(contour) >= 3:
				eps = 0.01 * cv2.arcLength(contour, True)
				simp = cv2.approxPolyDP(contour, eps, True)
				if len(simp) >= 3:
					coords = simp.reshape(-1, 2)
					try:
						poly = Polygon(coords)
						if poly.is_valid and poly.area > 20:  # small min area
							polygons.append(poly)
					except Exception:
						continue
		return polygons

	def merge_overlapping_polygons(self, polys: List[Polygon], overlap_threshold: float = 0.1) -> List[Polygon]:
		if not polys:
			return []
		merged: List[Polygon] = []
		remaining = polys.copy()
		while remaining:
			current = remaining.pop(0)
			to_merge = [current]
			i = 0
			while i < len(remaining):
				inter = current.intersection(remaining[i]).area
				union = current.union(remaining[i]).area
				if union > 0 and inter / union > overlap_threshold:
					to_merge.append(remaining.pop(i))
				else:
					i += 1
			union_geom: BaseGeometry = unary_union(to_merge)
			if hasattr(union_geom, "geoms"):
				for g in list(getattr(union_geom, "geoms")):
					if isinstance(g, Polygon):
						merged.append(g)
			else:
				if isinstance(union_geom, Polygon):
					merged.append(union_geom)
		return merged

