from typing import List

# Use cloud-compatible OpenCV
try:
    import cv2
except ImportError:
    from .cv2_cloud_compat import cv2
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
		
		# Create a copy of the list to avoid modifying the input
		merged: List[Polygon] = []
		remaining = list(polys)  # Explicit copy for clarity
		
		while remaining:
			current = remaining.pop(0)
			to_merge = [current]
			i = 0
			while i < len(remaining):
				# Optimize: Check bounding box overlap first (cheaper than actual intersection)
				current_bounds = current.bounds
				candidate_bounds = remaining[i].bounds
				
				# Quick rejection test using bounding boxes
				if (current_bounds[2] < candidate_bounds[0] or  # current max_x < candidate min_x
					current_bounds[0] > candidate_bounds[2] or  # current min_x > candidate max_x
					current_bounds[3] < candidate_bounds[1] or  # current max_y < candidate min_y
					current_bounds[1] > candidate_bounds[3]):   # current min_y > candidate max_y
					i += 1
					continue
				
				# Only compute expensive intersection if bounding boxes overlap
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

