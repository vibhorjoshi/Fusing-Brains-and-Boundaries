from typing import Dict

# Use cloud-compatible OpenCV
try:
    import cv2
except ImportError:
    from .cv2_cloud_compat import cv2
import numpy as np


class HybridRegularizer:
	"""Produce three regularized variants of a mask: RT, RR, FER (demo versions)."""

	def __init__(self, config):
		self.config = config

	def apply(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
		# Ensure binary float mask
		m = (mask > 0.5).astype(np.float32)

		# RT: mild closing to straighten boundaries a bit
		kernel_rt = np.ones((3, 3), np.uint8)
		rt = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel_rt, iterations=1).astype(np.float32)

		# RR: opening then closing to remove noise and fill small gaps
		kernel_rr = np.ones((5, 5), np.uint8)
		rr_tmp = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kernel_rr, iterations=1)
		rr = cv2.morphologyEx(rr_tmp, cv2.MORPH_CLOSE, kernel_rr, iterations=1).astype(np.float32)

		# FER: edge-aware dilation then threshold
		edges = cv2.Canny((m * 255).astype(np.uint8), 50, 150)
		dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
		fer = ((dilated > 0) | (m > 0.5)).astype(np.float32)

		return {
			"original": m,
			"rt": rt,
			"rr": rr,
			"fer": fer,
		}

