"""
RL-based Adaptive Fusion for Building Footprints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

class QNetwork(nn.Module):
    """Q-network for RL-based fusion"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from buffer"""
        return [self.buffer[i] for i in np.random.choice(len(self.buffer), batch_size)]
    
    def __len__(self) -> int:
        return len(self.buffer)

class AdaptiveFusion:
    """RL-based Adaptive Fusion for building footprints"""
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # Define RL parameters
        self.input_dim = 12  # Features from original and regularized masks
        self.output_dim = 4  # Actions: select original, RT, RR, or FER
        
        # Create Q-network
        self.q_network = QNetwork(self.input_dim, self.output_dim).to(self.device)
        self.target_network = QNetwork(self.input_dim, self.output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def extract_features(self, mask: np.ndarray) -> np.ndarray:
        """Extract features from mask for RL state"""
        # Simple features
        area = np.sum(mask) / mask.size
        
        # Shape complexity (perimeter / sqrt(area))
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        shape_complexity = 0.0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            contour_area = cv2.contourArea(largest_contour)
            if contour_area > 0:
                shape_complexity = perimeter / np.sqrt(contour_area)
        
        # Return features as array
        return np.array([
            area,
            shape_complexity,
            np.mean(mask),
        ])
    
    def calculate_reward(self, selected_mask: np.ndarray, gt_mask: np.ndarray = None) -> float:
        """Calculate reward for RL agent (IoU with ground truth if available)"""
        if gt_mask is not None:
            # IoU with ground truth
            intersection = np.logical_and(selected_mask, gt_mask).sum()
            union = np.logical_or(selected_mask, gt_mask).sum()
            iou = intersection / (union + 1e-6)
            return float(iou)
        else:
            # When no ground truth, reward based on mask quality heuristics
            contours, _ = cv2.findContours(
                (selected_mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return 0.0
                
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Circularity (4π*area/perimeter²) - rewards more compact shapes
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
            
            # Reward = normalized circularity (0 to 1)
            return min(max(circularity, 0.0), 1.0)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            # Random exploration
            return np.random.randint(0, self.output_dim)
        else:
            # Greedy exploitation
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def update_model(self):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([s[0] for s in batch]).to(self.device)
        actions = torch.LongTensor([[s[1]] for s in batch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        next_states = torch.FloatTensor([s[3] for s in batch]).to(self.device)
        
        # Calculate current Q values
        q_values = self.q_network(states).gather(1, actions)
        
        # Calculate next Q values with target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values)
        
        # Calculate loss
        loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.if optimizer is not None: optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network occasionally
        if np.random.rand() < 0.01:  # 1% chance each update
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, regularized_results: List[Dict[str, np.ndarray]], iterations: int = 50) -> List[float]:
        """Train the RL agent on regularized masks"""
        rewards = []
        
        for _ in range(iterations):
            batch_rewards = []
            
            for result in regularized_results:
                # Extract features for each variant
                original_features = self.extract_features(result["original"])
                rt_features = self.extract_features(result["rt"])
                rr_features = self.extract_features(result["rr"])
                fer_features = self.extract_features(result["fer"])
                
                # Combine features into state
                state = np.concatenate([original_features, rt_features, rr_features, fer_features])
                
                # Select action
                action = self.select_action(state)
                
                # Execute action and get mask
                masks = [result["original"], result["rt"], result["rr"], result["fer"]]
                selected_mask = masks[action]
                
                # Calculate reward
                ground_truth = result.get("ground_truth")
                reward = self.calculate_reward(selected_mask, ground_truth)
                batch_rewards.append(reward)
                
                # Store experience in replay buffer (no next state in this simplified case)
                self.replay_buffer.push(state, action, reward, state)
                
                # Update model
                self.update_model()
            
            # Track average reward per iteration
            rewards.append(np.mean(batch_rewards))
        
        return rewards
    
    def fuse_masks(self, regularized_results: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
        """Apply RL-based fusion to select best mask for each building"""
        fused_results = []
        
        for result in regularized_results:
            # Extract features for each variant
            original_features = self.extract_features(result["original"])
            rt_features = self.extract_features(result["rt"])
            rr_features = self.extract_features(result["rr"])
            fer_features = self.extract_features(result["fer"])
            
            # Combine features into state
            state = np.concatenate([original_features, rt_features, rr_features, fer_features])
            
            # Get Q-values for each action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Select best action based on Q-values
            action = np.argmax(q_values)
            
            # Get corresponding mask
            masks = [result["original"], result["rt"], result["rr"], result["fer"]]
            method_names = ["original", "rt", "rr", "fer"]
            selected_mask = masks[action]
            selected_method = method_names[action]
            
            # Create result
            fused_result = {
                "original": result["original"],
                "fused": selected_mask,
                "selected_method": selected_method,
                "q_values": q_values.tolist(),
                "ground_truth": result.get("ground_truth")
            }
            
            fused_results.append(fused_result)
        
        return fused_results
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # Continue with freshly initialized model