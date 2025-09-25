from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNNTrainer:
	def __init__(self, config):
		self.config = config
		self.model = None

	def create_model(self, num_classes: int = 2):
		model = maskrcnn_resnet50_fpn(weights=None)
		# Replace heads - check if model has the expected structure
		try:
			in_features = getattr(getattr(model.roi_heads.box_predictor, "cls_score", object()), "in_features", 1024)
			model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
		except AttributeError:
			# Fallback for different model structure
			in_features = 1024  # default
			model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
		
		try:
			in_features_mask = getattr(getattr(model.roi_heads.mask_predictor, "conv5_mask", object()), "in_channels", 256)
			hidden_layer = 256
			model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
		except AttributeError:
			# Fallback for different model structure
			hidden_layer = 256
			in_features_mask = 256  # default
			model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
		
		self.model = model

	def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List[float], List[float]]:
		if self.model is None:
			self.create_model()
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(device)
		params = [p for p in self.model.parameters() if p.requires_grad]
		optimizer = torch.optim.AdamW(params, lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)

		train_losses: List[float] = []
		val_ious: List[float] = []

		for epoch in range(self.config.NUM_EPOCHS):
			self.model.train()
			epoch_loss = 0.0
			for images, targets in train_loader:
				images = [img.to(device) for img in images]
				targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
				loss_dict = self.model(images, targets)
				losses = sum(loss for loss in loss_dict.values())
				
    if optimizer is not None: optimizer.zero_grad()
				# Check if losses is a tensor
				if hasattr(losses, 'backward') and hasattr(losses, 'item'):
					losses.backward()
					epoch_loss += float(losses.item())
				else:
					# Fallback for non-tensor losses
					epoch_loss += float(losses) if isinstance(losses, (int, float)) else 0.0
				optimizer.step()
			train_losses.append(epoch_loss / max(1, len(train_loader)))

			# quick val IoU proxy using predicted masks on a few samples
			self.model.eval()
			ious = []
			with torch.no_grad():
				for images, targets in val_loader:
					images = [img.to(device) for img in images]
					outputs = self.model(images)
					for out, tgt in zip(outputs, targets):
						if len(out.get("masks", [])) == 0:
							ious.append(0.0)
							continue
						pred = (out["masks"][0, 0] > 0.5).cpu().numpy()
						gt = (tgt["masks"][0] > 0).cpu().numpy()
						inter = (pred & gt).sum()
						union = (pred | gt).sum()
						iou = inter / (union + 1e-8) if union > 0 else 0.0
						ious.append(float(iou))
					break  # limit for speed
			val_ious.append(sum(ious) / max(1, len(ious)))
		return train_losses, val_ious

