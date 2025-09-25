from collections import deque
import random
from typing import Dict, Tuple

# Use cloud-compatible OpenCV
try:
    import cv2
except ImportError:
    from .cv2_cloud_compat import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(nn.Module):
	"""Deep Q-Network for adaptive regularization fusion."""

	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
		super().__init__()
		self.fc1 = nn.Linear(state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, hidden_dim)
		self.fc4 = nn.Linear(hidden_dim, action_dim)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.relu(self.fc1(x))
		x = self.dropout(x)
		x = torch.relu(self.fc2(x))
		x = self.dropout(x)
		x = torch.relu(self.fc3(x))
		x = self.fc4(x)
		return x


class ReplayMemory:
	"""Replay buffer for DQN training."""

	def __init__(self, capacity: int):
		self.capacity = capacity
		self.memory = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def sample(self, batch_size: int):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class AdaptiveFusion:
	"""Reinforcement Learning-based adaptive fusion of regularization methods.

	Exposes:
	- extract_features(reg_outputs)
	- select_action(state, training=True)
	- fuse_masks(reg_outputs, action)
	- compute_reward(fused_mask, ground_truth)
	- train_step(), update_target_network(), decay_epsilon()
	"""

	def __init__(self, config):
		self.config = config
		self.state_dim = 12  # 4 features per stream (RT, RR, FER)
		self.action_dim = 27  # 3^3 combos of weights {0.0, 0.5, 1.0}

		# Networks
		self.q_network = DQNAgent(self.state_dim, self.action_dim).to(device)
		self.target_network = DQNAgent(self.state_dim, self.action_dim).to(device)
		self.target_network.load_state_dict(self.q_network.state_dict())

		# Optimizer and memory
		self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.RL_LEARNING_RATE)
		self.memory = ReplayMemory(config.RL_MEMORY_SIZE)

		# Exploration
		self.epsilon = config.RL_EPSILON_START
		self.epsilon_end = config.RL_EPSILON_END
		self.epsilon_decay = config.RL_EPSILON_DECAY

		# Precompute mapping: action -> (w_rt, w_rr, w_fer)
		self.action_to_weights: Dict[int, Tuple[float, float, float]] = self._create_action_mapping()

	def _create_action_mapping(self) -> Dict[int, Tuple[float, float, float]]:
		weights = [0.0, 0.5, 1.0]
		mapping: Dict[int, Tuple[float, float, float]] = {}
		idx = 0
		for w_rt in weights:
			for w_rr in weights:
				for w_fer in weights:
					total = w_rt + w_rr + w_fer
					if total > 0:
						mapping[idx] = (w_rt / total, w_rr / total, w_fer / total)
					else:
						mapping[idx] = (0.33, 0.33, 0.34)
					idx += 1
		return mapping

	def _geom_features_from_mask(self, mask: np.ndarray) -> Tuple[float, float, float, float]:
		contours, _ = cv2.findContours((mask.astype(np.float32) > 0.5).astype(np.uint8),
									   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			largest = max(contours, key=cv2.contourArea)
			area = cv2.contourArea(largest)
			h, w = mask.shape[:2]
			area_n = area / (h * w + 1e-6)

			perim = cv2.arcLength(largest, True)
			perim_n = perim / (2 * (h + w) + 1e-6)

			x, y, bw, bh = cv2.boundingRect(largest)
			rect_area = bw * bh
			rectangularity = area / (rect_area + 1e-6) if rect_area > 0 else 0.0

			hull = cv2.convexHull(largest)
			hull_area = cv2.contourArea(hull)
			convexity = area / (hull_area + 1e-6) if hull_area > 0 else 0.0
			return area_n, perim_n, rectangularity, convexity
		return 0.0, 0.0, 0.0, 0.0

	def extract_features(self, reg_outputs: Dict[str, np.ndarray], ground_truth=None) -> np.ndarray:
		feats = []
		for key in ["rt", "rr", "fer"]:
			feats.extend(self._geom_features_from_mask(reg_outputs[key]))
		return np.array(feats, dtype=np.float32)

	def select_action(self, state: np.ndarray, training: bool = True) -> int:
		if training and random.random() < self.epsilon:
			return random.randint(0, self.action_dim - 1)
		with torch.no_grad():
			s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
			q = self.q_network(s)
			return int(q.argmax(dim=1).item())

	def fuse_masks(self, reg_outputs: Dict[str, np.ndarray], action: int) -> np.ndarray:
		w_rt, w_rr, w_fer = self.action_to_weights[action]
		fused = w_rt * reg_outputs["rt"] + w_rr * reg_outputs["rr"] + w_fer * reg_outputs["fer"]
		return (fused > 0.5).astype(np.float32)

	@staticmethod
	def _basic_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
		pred_b = pred.astype(bool)
		gt_b = gt.astype(bool)
		inter = np.logical_and(pred_b, gt_b).sum()
		union = np.logical_or(pred_b, gt_b).sum()
		iou = inter / (union + 1e-8) if union > 0 else 0.0
		precision = inter / (pred_b.sum() + 1e-8)
		recall = inter / (gt_b.sum() + 1e-8)
		f1 = 2 * precision * recall / (precision + recall + 1e-8)
		return iou, precision, recall, f1

	def compute_reward(self, fused_mask: np.ndarray, ground_truth: np.ndarray) -> float:
		if ground_truth is None:
			return 0.0
		iou, precision, recall, f1 = self._basic_metrics(fused_mask, ground_truth)
		return 0.6 * iou + 0.3 * f1 + 0.1 * precision

	def train_step(self) -> float:
		if len(self.memory) < self.config.RL_BATCH_SIZE:
			return 0.0

		batch = self.memory.sample(self.config.RL_BATCH_SIZE)
		states, actions, rewards, next_states, dones = zip(*batch)

		states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
		next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
		actions = torch.tensor(actions, dtype=torch.long, device=device)
		rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
		# Ensure 'dones' is a tensor of bools regardless of numpy scalar types
		dones = torch.tensor([bool(d) for d in dones], dtype=torch.bool, device=device)
		q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
		next_q = self.target_network(next_states).max(dim=1)[0]
		target = rewards + (0.99 * next_q * (~dones))

		loss = nn.MSELoss()(q_values, target.detach())
		if self.optimizer is not None:
			self.optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
		if self.optimizer is not None:
			self.optimizer.step()
		return float(loss.item())

	def update_target_network(self):
		self.target_network.load_state_dict(self.q_network.state_dict())

	def decay_epsilon(self):
		self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

