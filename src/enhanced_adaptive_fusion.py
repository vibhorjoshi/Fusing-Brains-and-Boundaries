"""
Enhanced Adaptive Fusion with Learned Proposals

This module implements an improved GPU-accelerated adaptive fusion model that:
1. Incorporates Mask R-CNN logits/probability maps as additional streams 
2. Uses image-conditioned features and CNN embeddings for richer state representation
3. Expands the action space to continuous weights using policy gradient methods
4. Scales to larger datasets with batched processing across multiple states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Normal

import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Union, Any

# Feature extraction backbone
class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for image-conditioned embeddings."""
    
    def __init__(self, in_channels: int = 3, out_features: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature embedding tensor [B, out_features]
        """
        features = self.backbone(x)
        return self.fc(features)

# Actor-Critic Network for Continuous Action Space
class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for policy gradient methods with continuous action space."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Actor (policy) network outputs mean and log_std for each action dimension
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: State tensor [B, state_dim]
            
        Returns:
            Tuple of (action_mean, action_log_std, value)
        """
        shared_features = self.shared(x)
        
        # Actor output: action mean and log_std
        action_mean = self.actor_mean(shared_features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic output: state value
        value = self.critic(shared_features)
        
        return action_mean, action_log_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy distribution.
        
        Args:
            state: State tensor [B, state_dim]
            deterministic: If True, return mean action without sampling
            
        Returns:
            Tuple of (sampled_action, log_prob, entropy)
        """
        action_mean, action_log_std, value = self(state)
        action_std = torch.exp(action_log_std)
        
        if deterministic:
            # During evaluation/inference, use the mean action
            return action_mean, None, None
        
        # Create Normal distribution and sample
        normal = Normal(action_mean, action_std)
        action = normal.rsample()  # Reparameterized sampling for backprop
        log_prob = normal.log_prob(action).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        # Apply sigmoid to constrain actions between 0 and 1
        action = torch.sigmoid(action)
        
        return action, log_prob, entropy

# Enhanced Experience Replay Buffer
class EnhancedReplayMemory:
    """GPU-optimized prioritized replay buffer for PPO training."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device="cuda"):
        self.capacity = capacity
        self.device = device
        
        # Pre-allocate tensors for efficient storage
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.size = 0

    def push(self, 
             state: torch.Tensor, 
             action: torch.Tensor, 
             reward: torch.Tensor, 
             next_state: torch.Tensor, 
             done: torch.Tensor,
             log_prob: Optional[torch.Tensor] = None):
        """Add experience to replay buffer."""
        idx = self.ptr
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        if log_prob is not None:
            self.log_probs[idx] = log_prob
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'log_probs': self.log_probs[indices] if hasattr(self, 'log_probs') else None
        }
        
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all stored transitions (used for PPO update)."""
        return {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_states': self.next_states[:self.size],
            'dones': self.dones[:self.size],
            'log_probs': self.log_probs[:self.size] if hasattr(self, 'log_probs') else None
        }

    def __len__(self):
        """Get current buffer size."""
        return self.size
    
    def clear(self):
        """Reset the replay buffer."""
        self.ptr = 0
        self.size = 0


class EnhancedAdaptiveFusion:
    """Enhanced adaptive fusion using Mask R-CNN proposals and PPO-based policy gradient methods.
    
    Key enhancements:
    1. Uses CNN feature extractor to incorporate image context
    2. Includes Mask R-CNN logits/probability maps as additional fusion streams
    3. Uses continuous action space with PPO algorithm
    4. Enriched state representation with overlap statistics
    5. Optimized for GPU training with mixed precision
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature dimensions
        self.image_feature_dim = config.get("IMAGE_FEATURE_DIM", 128)  # CNN embedding size
        self.reg_feature_dim = 5  # Features per regularization stream (expanded)
        self.num_reg_streams = 3  # RT, RR, FER
        self.proposal_feature_dim = 4  # Features from Mask R-CNN proposals
        
        # Calculate total state dimension
        self.state_dim = (
            self.image_feature_dim +                       # Image CNN features
            self.reg_feature_dim * self.num_reg_streams +  # Regularization features
            self.proposal_feature_dim +                    # Proposal features
            9                                              # Overlap statistics (3x3 matrix)
        )
        
        # Continuous action space: weight for each stream including Mask R-CNN proposals
        self.action_dim = 4  # Weights for RT, RR, FER, and Mask R-CNN proposals
        
        # Feature extraction networks
        self.feature_extractor = CNNFeatureExtractor(
            in_channels=1,  # Grayscale images
            out_features=self.image_feature_dim
        ).to(self.device)
        
        # Policy network (Actor-Critic)
        self.policy = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=config.get("RL_HIDDEN_DIM", 256)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.feature_extractor.parameters()),
            lr=config.RL_LEARNING_RATE,
            weight_decay=config.get("RL_WEIGHT_DECAY", 1e-4)
        )
        
        # PPO hyperparameters
        self.ppo_epochs = config.get("PPO_EPOCHS", 10)
        self.ppo_clip = config.get("PPO_CLIP", 0.2)
        self.value_coef = config.get("VALUE_COEF", 0.5)
        self.entropy_coef = config.get("ENTROPY_COEF", 0.01)
        self.gamma = config.get("RL_GAMMA", 0.99)
        self.gae_lambda = config.get("GAE_LAMBDA", 0.95)
        self.batch_size = config.get("RL_BATCH_SIZE", 64)
        
        # Replay memory for collecting experiences within an episode
        self.memory = EnhancedReplayMemory(
            capacity=config.RL_MEMORY_SIZE,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Mixed precision training
        self.use_amp = hasattr(torch.cuda, 'amp') and config.get("USE_MIXED_PRECISION", True)
        self.scaler = GradScaler() if self.use_amp else None
        
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CNN features from input images.
        
        Args:
            images: Input image tensor [B, 1, H, W]
            
        Returns:
            Image feature embedding [B, image_feature_dim]
        """
        with torch.no_grad():
            return self.feature_extractor(images)
            
    def _compute_overlap_stats(self, masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute pairwise overlap statistics between different mask streams.
        
        Args:
            masks: Dictionary of mask tensors
            
        Returns:
            Tensor of overlap statistics [B, 9]
        """
        batch_size = masks["original"].shape[0]
        stats = torch.zeros((batch_size, 9), device=self.device)
        
        # List of mask keys to compare
        keys = ["rt", "rr", "fer", "proposal"]
        
        # Calculate IoU and other overlap metrics for each pair
        idx = 0
        for i, key1 in enumerate(keys):
            if key1 not in masks:
                continue
                
            mask1 = masks[key1]
            for j, key2 in enumerate(keys[i:]):
                if key2 not in masks:
                    continue
                    
                mask2 = masks[key2]
                
                # Calculate intersection and union
                intersection = torch.sum(mask1 * mask2, dim=(1, 2, 3))
                union = torch.sum(torch.clamp(mask1 + mask2, 0, 1), dim=(1, 2, 3))
                
                # IoU
                iou = intersection / (union + 1e-6)
                
                # Store in stats tensor (if we have space)
                if idx < 9:
                    stats[:, idx] = iou
                    idx += 1
                    
        return stats
            
    def _extract_mask_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract enhanced geometric features from a mask.
        
        Args:
            mask: Binary mask tensor [B, 1, H, W]
            
        Returns:
            Feature tensor [B, 5]
        """
        batch_size = mask.shape[0]
        features = torch.zeros((batch_size, self.reg_feature_dim), device=self.device)
        
        # 1. Area ratio
        area = torch.sum(mask, dim=(1, 2, 3))
        total_area = mask.shape[2] * mask.shape[3]
        features[:, 0] = area / total_area
        
        # 2. Edge density (approximation using gradient magnitude)
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=self.device).float()
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=self.device).float()
        
        edges_x = F.conv2d(mask, sobel_x, padding=1)
        edges_y = F.conv2d(mask, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        edge_density = torch.sum(edge_magnitude > 0.1, dim=(1, 2, 3)) / (total_area + 1e-6)
        features[:, 1] = edge_density
        
        # 3. Compactness (using area and perimeter approximation)
        perimeter = torch.sum(edge_magnitude > 0.1, dim=(1, 2, 3))
        compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        features[:, 2] = compactness
        
        # 4. Connectedness (approximated using connected component count)
        # This is an approximation as true connected component analysis is not easily done in batched tensors
        # We use a simplified version based on edge density
        features[:, 3] = 1.0 / (1.0 + edge_density)
        
        # 5. Mean confidence
        features[:, 4] = torch.mean(mask, dim=(1, 2, 3))
        
        return features
        
    def extract_features(self, reg_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract comprehensive features for RL state representation.
        
        Args:
            reg_outputs: Dictionary of tensors including:
                - 'original': Original image [B, 1, H, W]
                - 'rt', 'rr', 'fer': Regularized masks [B, 1, H, W]
                - 'proposal': Mask R-CNN proposal [B, 1, H, W]
                - 'logits': Mask R-CNN logits [B, C, H, W] (optional)
                
        Returns:
            State tensor [B, state_dim]
        """
        batch_size = reg_outputs["original"].shape[0]
        state = torch.zeros((batch_size, self.state_dim), device=self.device)
        
        # 1. Extract image features from original image
        image_features = self.extract_image_features(reg_outputs["original"])
        state[:, :self.image_feature_dim] = image_features
        
        # 2. Extract features from each regularization mask
        offset = self.image_feature_dim
        for i, key in enumerate(["rt", "rr", "fer"]):
            if key in reg_outputs:
                mask_features = self._extract_mask_features(reg_outputs[key])
                feature_idx = offset + i * self.reg_feature_dim
                state[:, feature_idx:feature_idx + self.reg_feature_dim] = mask_features
        
        # 3. Extract features from Mask R-CNN proposal
        proposal_offset = offset + self.num_reg_streams * self.reg_feature_dim
        if "proposal" in reg_outputs:
            proposal_features = self._extract_mask_features(reg_outputs["proposal"])[:, :self.proposal_feature_dim]
            state[:, proposal_offset:proposal_offset + self.proposal_feature_dim] = proposal_features
        
        # 4. Compute overlap statistics
        overlap_offset = proposal_offset + self.proposal_feature_dim
        overlap_stats = self._compute_overlap_stats(reg_outputs)
        state[:, overlap_offset:] = overlap_stats
        
        return state
        
    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Select fusion weights using policy network.
        
        Args:
            state: State tensor [B, state_dim]
            training: Whether to use exploration and store log_probs
            
        Returns:
            Tuple of (actions, log_probs, entropy)
        """
        # Normalize state
        state_mean = state.mean(dim=0, keepdim=True)
        state_std = state.std(dim=0, keepdim=True) + 1e-6
        normalized_state = (state - state_mean) / state_std
        
        # Get action from policy
        if training:
            actions, log_probs, entropy = self.policy.get_action(normalized_state)
            return actions, log_probs, entropy
        else:
            actions, _, _ = self.policy.get_action(normalized_state, deterministic=True)
            return actions, None, None
        
    def fuse_masks(self, 
                  reg_outputs: Dict[str, torch.Tensor], 
                  actions: torch.Tensor) -> torch.Tensor:
        """Fuse masks based on continuous action weights.
        
        Args:
            reg_outputs: Dictionary of regularized masks [B,1,H,W]
            actions: Weight tensor [B,4] with values in [0,1]
            
        Returns:
            Fused masks [B,1,H,W]
        """
        batch_size = actions.shape[0]
        
        # Get normalized weights (sum to 1.0)
        weights_sum = actions.sum(dim=1, keepdim=True) + 1e-6
        normalized_weights = actions / weights_sum
        
        # Extract regularized masks
        rt_masks = reg_outputs.get("rt")  # [B,1,H,W]
        rr_masks = reg_outputs.get("rr")  # [B,1,H,W]
        fer_masks = reg_outputs.get("fer")  # [B,1,H,W]
        proposal_masks = reg_outputs.get("proposal")  # [B,1,H,W]
        
        # Reshape weights for broadcasting
        w_rt = normalized_weights[:, 0].view(batch_size, 1, 1, 1)
        w_rr = normalized_weights[:, 1].view(batch_size, 1, 1, 1)
        w_fer = normalized_weights[:, 2].view(batch_size, 1, 1, 1)
        w_proposal = normalized_weights[:, 3].view(batch_size, 1, 1, 1)
        
        # Initialize with zeros
        fused_masks = torch.zeros_like(reg_outputs["original"])
        
        # Add each component if available
        if rt_masks is not None:
            fused_masks = fused_masks + w_rt * rt_masks
        if rr_masks is not None:
            fused_masks = fused_masks + w_rr * rr_masks
        if fer_masks is not None:
            fused_masks = fused_masks + w_fer * fer_masks
        if proposal_masks is not None:
            fused_masks = fused_masks + w_proposal * proposal_masks
        
        # Threshold to get binary mask
        binary_masks = (fused_masks > 0.5).float()
        
        return binary_masks
        
    def compute_reward(self, 
                       fused_masks: torch.Tensor, 
                       ground_truth: torch.Tensor) -> torch.Tensor:
        """Compute enhanced rewards for the fusion results.
        
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
        
        # Calculate precision and recall
        precision = intersection / (torch.sum(fused, dim=(1, 2, 3)) + 1e-6)
        recall = intersection / (torch.sum(gt, dim=(1, 2, 3)) + 1e-6)
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Boundary F1 score approximation using edge detection
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=self.device).float()
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=self.device).float()
        
        # Edge detection for predictions and ground truth
        pred_edges_x = F.conv2d(fused, sobel_x, padding=1)
        pred_edges_y = F.conv2d(fused, sobel_y, padding=1)
        pred_edges = (torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2) > 0.1).float()
        
        gt_edges_x = F.conv2d(gt, sobel_x, padding=1)
        gt_edges_y = F.conv2d(gt, sobel_y, padding=1)
        gt_edges = (torch.sqrt(gt_edges_x ** 2 + gt_edges_y ** 2) > 0.1).float()
        
        # Calculate boundary IoU
        boundary_intersection = torch.sum(pred_edges * gt_edges, dim=(1, 2, 3))
        boundary_union = torch.sum(torch.clamp(pred_edges + gt_edges, 0, 1), dim=(1, 2, 3))
        boundary_iou = boundary_intersection / (boundary_union + 1e-6)
        
        # Combined reward with emphasis on boundary accuracy
        rewards = 0.5 * iou + 0.3 * f1 + 0.2 * boundary_iou
        
        # Scale to more intuitive range
        scaled_rewards = rewards * 100.0
        
        return scaled_rewards
        
    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of rewards [T]
            values: Tensor of value predictions [T]
            dones: Tensor of done flags [T]
            
        Returns:
            Tuple of (returns, advantages)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        next_value = 0
        next_advantage = 0
        
        for t in reversed(range(len(rewards))):
            # Calculate returns (discounted sum of rewards)
            returns[t] = rewards[t] + self.gamma * next_value * (1.0 - dones[t])
            
            # TD error
            td_error = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            
            # Generalized Advantage Estimation
            advantages[t] = td_error + self.gamma * self.gae_lambda * next_advantage * (1.0 - dones[t])
            
            next_value = values[t]
            next_advantage = advantages[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        
        return returns, advantages
        
    def update_policy(self, states: torch.Tensor, actions: torch.Tensor, 
                     log_probs_old: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> Dict[str, float]:
        """Update policy using PPO algorithm.
        
        Args:
            states: Tensor of states [B, state_dim]
            actions: Tensor of actions [B, action_dim]
            log_probs_old: Tensor of old log probs [B]
            returns: Tensor of returns [B]
            advantages: Tensor of advantages [B]
            
        Returns:
            Dictionary of loss metrics
        """
        # Normalize states for numerical stability
        state_mean = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True) + 1e-6
        normalized_states = (states - state_mean) / state_std
        
        # Optimize for multiple epochs
        loss_metrics = {"actor_loss": 0, "critic_loss": 0, "entropy": 0}
        
        for _ in range(self.ppo_epochs):
            # Get random permutation of indices for mini-batches
            indices = torch.randperm(states.shape[0])
            
            for start in range(0, states.shape[0], self.batch_size):
                # Get mini-batch
                end = start + self.batch_size
                if end > states.shape[0]:
                    end = states.shape[0]
                batch_indices = indices[start:end]
                
                if self.use_amp:
                    # Use mixed precision for training
                    with autocast():
                        # Get current policy distribution and values
                        action_mean, action_log_std, values = self.policy(normalized_states[batch_indices])
                        action_std = torch.exp(action_log_std)
                        
                        # Create Normal distribution
                        dist = Normal(action_mean, action_std)
                        
                        # Get log probs of actions under current policy
                        log_probs = dist.log_prob(actions[batch_indices]).sum(-1)
                        
                        # Calculate ratio and clipped ratio
                        ratio = torch.exp(log_probs - log_probs_old[batch_indices])
                        clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
                        
                        # Calculate actor loss
                        surrogate1 = ratio * advantages[batch_indices]
                        surrogate2 = clipped_ratio * advantages[batch_indices]
                        actor_loss = -torch.min(surrogate1, surrogate2).mean()
                        
                        # Calculate critic loss
                        critic_loss = F.mse_loss(values.squeeze(-1), returns[batch_indices])
                        
                        # Calculate entropy bonus
                        entropy = dist.entropy().mean()
                        
                        # Combined loss
                        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                    
                    # Optimize
                    self.if optimizer is not None: optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular computation (no mixed precision)
                    # Get current policy distribution and values
                    action_mean, action_log_std, values = self.policy(normalized_states[batch_indices])
                    action_std = torch.exp(action_log_std)
                    
                    # Create Normal distribution
                    dist = Normal(action_mean, action_std)
                    
                    # Get log probs of actions under current policy
                    log_probs = dist.log_prob(actions[batch_indices]).sum(-1)
                    
                    # Calculate ratio and clipped ratio
                    ratio = torch.exp(log_probs - log_probs_old[batch_indices])
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
                    
                    # Calculate actor loss
                    surrogate1 = ratio * advantages[batch_indices]
                    surrogate2 = clipped_ratio * advantages[batch_indices]
                    actor_loss = -torch.min(surrogate1, surrogate2).mean()
                    
                    # Calculate critic loss
                    critic_loss = F.mse_loss(values.squeeze(-1), returns[batch_indices])
                    
                    # Calculate entropy bonus
                    entropy = dist.entropy().mean()
                    
                    # Combined loss
                    loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                    
                    # Optimize
                    self.if optimizer is not None: optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Track metrics
                loss_metrics["actor_loss"] += actor_loss.item()
                loss_metrics["critic_loss"] += critic_loss.item()
                loss_metrics["entropy"] += entropy.item()
        
        # Average metrics
        num_batches = (states.shape[0] + self.batch_size - 1) // self.batch_size
        total_batches = num_batches * self.ppo_epochs
        for k in loss_metrics:
            loss_metrics[k] /= total_batches
        
        return loss_metrics
        
    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform a PPO update using collected experiences.
        
        Returns:
            Dictionary of training metrics or None if not enough data
        """
        # Check if enough data is available
        if len(self.memory) < self.batch_size:
            return None
            
        # Get all experiences from buffer
        batch = self.memory.get_all()
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        old_log_probs = batch["log_probs"]
        
        # Compute state values
        with torch.no_grad():
            _, _, values = self.policy(states)
            values = values.squeeze(-1)
        
        # Compute advantages and returns
        returns, advantages = self._compute_advantages(rewards, values, dones)
        
        # Update policy
        metrics = self.update_policy(states, actions, old_log_probs, returns, advantages)
        
        # Clear memory after update
        self.memory.clear()
        
        return metrics
        
    def save_model(self, path: str):
        """Save model weights to disk."""
        torch.save({
            'policy': self.policy.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path: str):
        """Load model weights from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
    def process_batch(self, 
                     reg_outputs: Dict[str, List[np.ndarray]], 
                     ground_truth: List[np.ndarray],
                     rcnn_proposals: List[np.ndarray] = None,
                     rcnn_logits: List[np.ndarray] = None,
                     training: bool = True) -> Tuple[List[np.ndarray], List[float]]:
        """Process a batch of regularized masks.
        
        Args:
            reg_outputs: Dictionary of regularized masks as numpy arrays
            ground_truth: List of ground truth masks as numpy arrays
            rcnn_proposals: List of Mask R-CNN proposal masks
            rcnn_logits: List of Mask R-CNN logit maps
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
                # Add batch and channel dimensions if needed
                if mask.ndim == 2:
                    tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
                elif mask.ndim == 3:
                    tensor = torch.from_numpy(mask).float().unsqueeze(0)
                else:
                    tensor = torch.from_numpy(mask).float()
                tensors.append(tensor)
            # Stack into batch
            tensor_outputs[reg_type] = torch.cat(tensors, dim=0).to(self.device)
            
        # Add Mask R-CNN proposals if available
        if rcnn_proposals is not None:
            proposal_tensors = []
            for prop in rcnn_proposals:
                if prop.ndim == 2:
                    tensor = torch.from_numpy(prop).float().unsqueeze(0).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(prop).float().unsqueeze(0)
                proposal_tensors.append(tensor)
            tensor_outputs["proposal"] = torch.cat(proposal_tensors, dim=0).to(self.device)
            
        # Add Mask R-CNN logits if available
        if rcnn_logits is not None:
            logit_tensors = []
            for logit in rcnn_logits:
                if logit.ndim == 2:
                    tensor = torch.from_numpy(logit).float().unsqueeze(0).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(logit).float()
                logit_tensors.append(tensor)
            tensor_outputs["logits"] = torch.cat(logit_tensors, dim=0).to(self.device)
            
        # Convert ground truth to tensor
        gt_tensors = []
        for gt in ground_truth:
            if gt.ndim == 2:
                tensor = torch.from_numpy((gt > 0.5).astype(np.float32)).float().unsqueeze(0).unsqueeze(0)
            else:
                tensor = torch.from_numpy((gt > 0.5).astype(np.float32)).float()
            gt_tensors.append(tensor)
        gt_batch = torch.cat(gt_tensors, dim=0).to(self.device)
        
        # Extract features
        state = self.extract_features(tensor_outputs)
        
        # Select actions
        if training:
            actions, log_probs, _ = self.select_action(state, training=True)
        else:
            actions, _, _ = self.select_action(state, training=False)
            log_probs = None
        
        # Fuse masks
        fused_masks = self.fuse_masks(tensor_outputs, actions)
        
        # Compute rewards
        rewards = self.compute_reward(fused_masks, gt_batch)
        
        # If training, store experiences
        if training:
            for i in range(batch_size):
                # Store experience
                self.memory.push(
                    state[i],
                    actions[i],
                    rewards[i],
                    state[i],  # use same state as next_state for simplicity
                    False,     # not terminal state
                    log_probs[i] if log_probs is not None else None
                )
        
        # Convert results back to numpy
        fused_np = [fused_masks[i, 0].cpu().numpy() for i in range(batch_size)]
        rewards_np = rewards.cpu().numpy().tolist()
        
        return fused_np, rewards_np