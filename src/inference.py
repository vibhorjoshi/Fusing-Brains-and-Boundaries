from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


class MaskRCNNInference:
	def __init__(self, model, config):
		self.model = model
		self.config = config
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)
		self.model.eval()

	def process_patch(self, patch: np.ndarray):
		# patch: CxHxW
		import torch
		with torch.no_grad():
			x = torch.from_numpy(patch).to(self.device)
			outputs = self.model([x])
			out = outputs[0]
			masks = []
			if "masks" in out and len(out["masks"]) > 0:
				for m in out["masks"]:
					masks.append((m[0] > 0.5).float().cpu().numpy())
		return masks, out

