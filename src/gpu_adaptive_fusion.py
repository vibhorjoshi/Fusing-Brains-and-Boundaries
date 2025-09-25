"""
GPU-Accelerated Adaptive Fusion for Building Footprint Regularization

This module implements a Deep Q-Network (DQN) based reinforcement learning approach
to adaptively fuse multiple regularization methods. The implementation is optimized
for GPU acceleration with batch processing and parallel computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional


class GPUDQNAgent(nn.Module):
    """Deep Q-Network for adaptive regularization fusion optimized for GPU.
    
    This model uses multiple hidden layers with batch normalization and
    dropout for improved stability and faster training on GPU.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Improved network architecture for GPU acceleration
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
        

class GPUReplayMemory:
    """GPU-optimized replay buffer for DQN training with tensor batch sampling."""

    def __init__(self, capacity: int, device="cuda"):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences and return as tensors on GPU."""
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to tensors and move to GPU for batch processing
        state_t = torch.FloatTensor(np.array(state)).to(self.device)
        action_t = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_t = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(np.array(next_state)).to(self.device)
        done_t = torch.FloatTensor(np.array(done, dtype=np.float32)).unsqueeze(1).to(self.device)
        
        return state_t, action_t, reward_t, next_state_t, done_t

    def __len__(self):
        """Get current buffer size."""
        return len(self.memory)


class GPUAdaptiveFusion:
    """Reinforcement Learning-based adaptive fusion of regularization methods
    with GPU acceleration and mixed precision training support.
    
    This class implements:
    - Batch processing of states and actions
    - Mixed precision training for improved performance
    - Parallelized reward computation
    - GPU-accelerated feature extraction
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model dimensions
        self.state_dim = 12  # 4 features per stream (RT, RR, FER)
        self.action_dim = 27  # 3^3 combos of weights {0.0, 0.5, 1.0}
        self.hidden_dim = config.get("RL_HIDDEN_DIM", 256)
        
        # GPU optimized networks
        self.q_network = GPUDQNAgent(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network = GPUDQNAgent(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and memory
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=config.RL_LEARNING_RATE,
                                   weight_decay=config.get("RL_WEIGHT_DECAY", 1e-4))
        self.memory = GPUReplayMemory(config.RL_MEMORY_SIZE, self.device)
        
        # Mixed precision training
        self.use_amp = hasattr(torch.cuda, 'amp') and config.get("USE_MIXED_PRECISION", True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Exploration parameters
        self.epsilon = config.RL_EPSILON_START
        self.epsilon_end = config.RL_EPSILON_END
        self.epsilon_decay = config.RL_EPSILON_DECAY
        self.batch_size = config.get("RL_BATCH_SIZE", 64)
        self.gamma = config.get("RL_GAMMA", 0.99)
        
        # Precompute action weight mapping
        self.action_to_weights: Dict[int, Tuple[float, float, float]] = self._create_action_mapping()
        self.weights_tensor = self._create_weights_tensor()
        
    def _create_action_mapping(self) -> Dict[int, Tuple[float, float, float]]:
        """Create a mapping from action index to regularization weights."""
        weights = [0.0, 0.5, 1.0]
        mapping = {}
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
        
    def _create_weights_tensor(self) -> torch.Tensor:
        """Create a tensor containing all possible weight combinations for batch processing."""
        weights_list = [list(weights) for weights in self.action_to_weights.values()]
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=self.device)
        return weights_tensor
        
    def extract_features(self, reg_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from regularized outputs for RL state representation.
        
        Args:
            reg_outputs: Dictionary of regularized masks on GPU
            
        Returns:
            Tensor of state features [B, state_dim] on GPU
        """
        batch_size = reg_outputs["original"].shape[0]
        features = torch.zeros((batch_size, self.state_dim), device=self.device)
        
        # Calculate features for each regularization type
        for i, reg_type in enumerate(["rt", "rr", "fer"]):
            if reg_type in reg_outputs:
                # Extract mask
                mask = reg_outputs[reg_type]
                original = reg_outputs["original"]
                
                # 1. Area ratio (masked area / total area)
                area_ratio = torch.sum(mask, dim=(1, 2, 3)) / (mask.shape[2] * mask.shape[3])
                
                # 2. Perimeter (approximation using edge detection)
                edges = torch.abs(F.conv2d(
                    mask,
                    torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], 
                               device=self.device),
                    padding=1
                ) > 0.1).float()
                perimeter = torch.sum(edges, dim=(1, 2, 3))
                
                # 3. IoU with original mask
                intersection = torch.sum(mask * original, dim=(1, 2, 3))
                union = torch.sum(torch.clamp(mask + original, 0, 1), dim=(1, 2, 3))
                iou = intersection / (union + 1e-6)
                
                # 4. Compactness (area / perimeter²)
                area = torch.sum(mask, dim=(1, 2, 3))
                compactness = area / (torch.pow(perimeter, 2) + 1e-6)
                
                # Store features for this regularization type
                feature_idx = i * 4
                features[:, feature_idx] = area_ratio
                features[:, feature_idx + 1] = perimeter / 1000.0  # Normalize
                features[:, feature_idx + 2] = iou
                features[:, feature_idx + 3] = compactness * 1000.0  # Normalize
        
        return features
        
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Select an action for a batch of states using epsilon-greedy policy.
        
        Args:
            state: Tensor of state features [B, state_dim]
            training: Whether to use epsilon-greedy exploration
            
        Returns:
            Tensor of selected actions [B]
        """
        batch_size = state.shape[0]
        
        if training and random.random() < self.epsilon:
            # Exploration: random actions
            return torch.randint(0, self.action_dim, (batch_size,), device=self.device)
        else:
            # Exploitation: select best action from Q-network
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values, dim=1)
                
    def fuse_masks(self, reg_outputs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        """Fuse regularized masks based on selected actions.
        
        Args:
            reg_outputs: Dictionary of regularized masks [B,1,H,W]
            actions: Tensor of selected actions [B]
            
        Returns:
            Fused masks [B,1,H,W]
        """
        batch_size = actions.shape[0]
        
        # Get weights for each selected action
        weights = torch.index_select(self.weights_tensor, 0, actions)
        
        # Extract regularized masks
        rt_masks = reg_outputs["rt"]  # [B,1,H,W]
        rr_masks = reg_outputs["rr"]  # [B,1,H,W]
        fer_masks = reg_outputs["fer"]  # [B,1,H,W]
        
        # Compute weighted sum for fusion (weights are [B,3])
        w_rt = weights[:, 0].view(batch_size, 1, 1, 1)
        w_rr = weights[:, 1].view(batch_size, 1, 1, 1)
        w_fer = weights[:, 2].view(batch_size, 1, 1, 1)
        
        # Weighted fusion
        fused_masks = w_rt * rt_masks + w_rr * rr_masks + w_fer * fer_masks
        
        # Threshold to get binary mask
        fused_masks = (fused_masks > 0.5).float()
        
        return fused_masks
        
    def compute_reward(self, fused_masks: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Compute rewards for the fusion results.
        
        Args:
            fused_masks: Tensor of fused masks [B,1,H,W]
            ground_truth: Tensor of ground truth masks [B,1,H,W]
            
        Returns:
            Tensor of rewards [B]
        """
        # Ensure binary masks
        fused = (fused_masks > 0.5).float()
        gt = (ground_truth > 0.5).float()
        
        # Calculate IoU (intersection over union)
        intersection = torch.sum(fused * gt, dim=(1, 2, 3))
        union = torch.sum(torch.clamp(fused + gt, 0, 1), dim=(1, 2, 3))
        iou = intersection / (union + 1e-6)
        
        # Scale IoU to reward range
        rewards = iou * 100.0
        
        return rewards
        
    def train_step(self) -> Optional[float]:
        """Perform a training step on a batch of experiences.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch from replay buffer
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        if self.use_amp:
            # Use mixed precision for faster computation
            with autocast():
                # Compute Q values for current state-action pairs
                q_values = self.q_network(state).gather(1, action)
                
                # Compute max Q values for next states (target network)
                with torch.no_grad():
                    next_q_values = self.target_network(next_state).max(1, keepdim=True)[0]
                
                # Compute target Q values
                target_q_values = reward + self.gamma * next_q_values * (1 - done)
                
                # Compute loss
                loss = F.smooth_l1_loss(q_values, target_q_values)
                
            # Scale gradients and optimize
            self.if optimizer is not None: optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular computation (no mixed precision)
            # Compute Q values for current state-action pairs
            q_values = self.q_network(state).gather(1, action)
            
            # Compute max Q values for next states (target network)
            with torch.no_grad():
                next_q_values = self.target_network(next_state).max(1, keepdim=True)[0]
            
            # Compute target Q values
            target_q_values = reward + self.gamma * next_q_values * (1 - done)
            
            # Compute loss
            loss = F.smooth_l1_loss(q_values, target_q_values)
            
            # Optimize
            self.if optimizer is not None: optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss.item()
        
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def save_model(self, path: str):
        """Save model weights to disk."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_model(self, path: str):
        """Load model weights from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        
    def process_batch(self, 
                     reg_outputs: Dict[str, List[np.ndarray]], 
                     ground_truth: List[np.ndarray],
                     training: bool = True) -> Tuple[List[np.ndarray], List[float]]:
        """Process a batch of regularized masks.
        
        Args:
            reg_outputs: Dictionary of regularized masks as numpy arrays
            ground_truth: List of ground truth masks as numpy arrays
            training: Whether to use exploration and update memory
            
        Returns:
            Tuple of (fused_masks, rewards)
        """
        batch_size = len(reg_outputs["original"])
        if batch_size == 0:
            return [], []
            
        # Convert numpy arrays to torch tensors
        tensor_outputs = {}
        for reg_type, masks in reg_outputs.items():
            tensors = []
            for mask in masks:
                # Add batch and channel dimensions
                tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
                tensors.append(tensor)
            # Stack into batch
            tensor_outputs[reg_type] = torch.cat(tensors, dim=0).to(self.device)
            
        # Convert ground truth to tensor
        gt_tensors = []
        for gt in ground_truth:
            tensor = torch.from_numpy((gt > 0.5).astype(np.float32)).float().unsqueeze(0).unsqueeze(0)
            gt_tensors.append(tensor)
        gt_batch = torch.cat(gt_tensors, dim=0).to(self.device)
        
        # Extract features
        state = self.extract_features(tensor_outputs)
        
        # Select actions
        actions = self.select_action(state, training)
        
        # Fuse masks
        fused_masks = self.fuse_masks(tensor_outputs, actions)
        
        # Compute rewards
        rewards = self.compute_reward(fused_masks, gt_batch)
        
        # If training, store experiences
        if training:
            for i in range(batch_size):
                # Get next state (we simulate it by slight perturbation)
                next_state = state[i].clone() + torch.randn_like(state[i]) * 0.01
                
                # Store experience
                self.memory.push(
                    state[i].cpu().numpy(),
                    actions[i].item(),
                    rewards[i].item(),
                    next_state.cpu().numpy(),
                    False  # not terminal state
                )
        
        # Convert results back to numpy
        fused_np = [fused_masks[i, 0].cpu().numpy() for i in range(batch_size)]
        rewards_np = rewards.cpu().numpy().tolist()
        
        return fused_np, rewards_np