from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import numpy as np
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import random
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import math
from concurrent.futures import ThreadPoolExecutor
import aiohttp

app = FastAPI(title='GeoAI Adaptive Fusion with RL Agent - Live World Map Testing', version='3.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ===============================
# ADAPTIVE FUSION RL AGENT
# ===============================

class DQNAgent(nn.Module):
    """Deep Q-Network for adaptive regularization fusion."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
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

class AdaptiveFusionRL:
    """Enhanced RL-based adaptive fusion for building detection with live world map testing."""
    
    def __init__(self):
        self.state_dim = 15  # Enhanced features: geometry (12) + texture (3)
        self.action_dim = 64  # More granular weight combinations
        
        # Networks
        self.q_network = DQNAgent(self.state_dim, self.action_dim).to(device)
        self.target_network = DQNAgent(self.state_dim, self.action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=0.001, weight_decay=1e-4)
        self.memory = ReplayMemory(50000)
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.gamma = 0.99
        
        # Action mapping for continuous weight space
        self.action_to_weights = self._create_enhanced_action_mapping()
        
        # Performance tracking
        self.performance_history = []
        self.global_metrics = {
            'total_processed': 0,
            'total_training_episodes': 0,
            'average_iou': 0.0,
            'best_iou': 0.0,
            'convergence_rate': 0.0,
            'adaptation_speed': 0.0
        }
        
    def _create_enhanced_action_mapping(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Create enhanced action mapping with 4-component weights (RT, RR, FER, MaskRCNN)"""
        mapping = {}
        idx = 0
        
        # Generate more granular weight combinations
        steps = 4  # 0.0, 0.33, 0.67, 1.0
        for w_rt in np.linspace(0, 1, steps):
            for w_rr in np.linspace(0, 1, steps):
                for w_fer in np.linspace(0, 1, steps):
                    for w_mask in np.linspace(0, 1, steps):
                        if idx < self.action_dim:
                            total = w_rt + w_rr + w_fer + w_mask
                            if total > 0:
                                mapping[idx] = (w_rt/total, w_rr/total, w_fer/total, w_mask/total)
                            else:
                                mapping[idx] = (0.25, 0.25, 0.25, 0.25)
                            idx += 1
        
        return mapping
    
    def extract_enhanced_features(self, 
                                reg_outputs: Dict[str, np.ndarray], 
                                satellite_image: np.ndarray = None,
                                location_coords: Tuple[float, float] = None) -> np.ndarray:
        """Extract enhanced features including geographic and contextual information."""
        
        features = []
        
        # Geometric features from each regularization method
        for key in ["rt", "rr", "fer"]:
            if key in reg_outputs:
                geom_feats = self._extract_geometric_features(reg_outputs[key])
                features.extend(geom_feats)
        
        # Texture and contextual features
        if satellite_image is not None:
            texture_feats = self._extract_texture_features(satellite_image)
            features.extend(texture_feats)
        else:
            features.extend([0.0, 0.0, 0.0])  # Placeholder
        
        return np.array(features, dtype=np.float32)
    
    def _extract_geometric_features(self, mask: np.ndarray) -> List[float]:
        """Extract geometric features from mask."""
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            h, w = mask.shape[:2]
            
            # Normalized area
            area_norm = area / (h * w + 1e-6)
            
            # Perimeter features
            perimeter = cv2.arcLength(largest, True)
            perimeter_norm = perimeter / (2 * (h + w) + 1e-6)
            
            # Compactness (isoperimetric ratio)
            compactness = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            
            # Rectangularity
            x, y, bw, bh = cv2.boundingRect(largest)
            rect_area = bw * bh
            rectangularity = area / (rect_area + 1e-6) if rect_area > 0 else 0.0
            
            return [area_norm, perimeter_norm, compactness, rectangularity]
        
        return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features from satellite image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Texture contrast (standard deviation)
        texture_contrast = np.std(gray) / 255.0
        
        # Local binary pattern approximation (simplified)
        lbp_variance = np.var(gray) / (255.0 * 255.0)
        
        return [edge_density, texture_contrast, lbp_variance]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def fuse_masks(self, 
                   reg_outputs: Dict[str, np.ndarray], 
                   action: int,
                   mask_rcnn_output: np.ndarray = None) -> np.ndarray:
        """Fuse masks using selected action weights."""
        
        w_rt, w_rr, w_fer, w_mask = self.action_to_weights[action]
        
        fused = (w_rt * reg_outputs.get("rt", np.zeros_like(list(reg_outputs.values())[0])) +
                 w_rr * reg_outputs.get("rr", np.zeros_like(list(reg_outputs.values())[0])) +
                 w_fer * reg_outputs.get("fer", np.zeros_like(list(reg_outputs.values())[0])))
        
        if mask_rcnn_output is not None:
            fused = fused + w_mask * mask_rcnn_output
        
        return (fused > 0.5).astype(np.float32)
    
    def compute_reward(self, fused_mask: np.ndarray, ground_truth: np.ndarray = None) -> float:
        """Compute reward for RL training."""
        if ground_truth is None:
            # Use quality heuristics when no ground truth available
            return self._compute_heuristic_reward(fused_mask)
        
        # IoU-based reward with additional metrics
        iou, precision, recall, f1 = self._compute_metrics(fused_mask, ground_truth)
        
        # Composite reward
        reward = 0.5 * iou + 0.2 * f1 + 0.15 * precision + 0.15 * recall
        
        # Bonus for high performance
        if iou > 0.8:
            reward += 0.1
        if f1 > 0.85:
            reward += 0.05
            
        return reward
    
    def _compute_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute IoU, precision, recall, F1."""
        pred_bool = pred > 0.5
        gt_bool = gt > 0.5
        
        intersection = np.logical_and(pred_bool, gt_bool).sum()
        union = np.logical_or(pred_bool, gt_bool).sum()
        
        iou = intersection / (union + 1e-8) if union > 0 else 0.0
        precision = intersection / (pred_bool.sum() + 1e-8)
        recall = intersection / (gt_bool.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return iou, precision, recall, f1
    
    def _compute_heuristic_reward(self, mask: np.ndarray) -> float:
        """Compute reward based on mask quality heuristics."""
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Reward for reasonable area
        area_reward = min(area / (mask.shape[0] * mask.shape[1]), 0.3)
        
        # Reward for shape regularity
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        shape_reward = min(compactness, 0.3)
        
        return area_reward + shape_reward
    
    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor([bool(d) for d in dones], dtype=torch.bool, device=device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

# Global RL agent
rl_agent = AdaptiveFusionRL()

# ===============================
# LIVE WORLD MAP DATA
# ===============================

WORLD_LOCATIONS = {
    # North America
    'new_york_usa': {'lat': 40.7128, 'lng': -74.0060, 'region': 'North America', 'population': 8175133},
    'los_angeles_usa': {'lat': 34.0522, 'lng': -118.2437, 'region': 'North America', 'population': 3971883},
    'toronto_canada': {'lat': 43.6532, 'lng': -79.3832, 'region': 'North America', 'population': 2794356},
    'mexico_city_mexico': {'lat': 19.4326, 'lng': -99.1332, 'region': 'North America', 'population': 9209944},
    
    # Europe  
    'london_uk': {'lat': 51.5074, 'lng': -0.1278, 'region': 'Europe', 'population': 9648110},
    'paris_france': {'lat': 48.8566, 'lng': 2.3522, 'region': 'Europe', 'population': 2161000},
    'berlin_germany': {'lat': 52.5200, 'lng': 13.4050, 'region': 'Europe', 'population': 3669491},
    'madrid_spain': {'lat': 40.4168, 'lng': -3.7038, 'region': 'Europe', 'population': 6642000},
    'rome_italy': {'lat': 41.9028, 'lng': 12.4964, 'region': 'Europe', 'population': 2873494},
    'amsterdam_netherlands': {'lat': 52.3676, 'lng': 4.9041, 'region': 'Europe', 'population': 873555},
    
    # Asia
    'tokyo_japan': {'lat': 35.6762, 'lng': 139.6503, 'region': 'Asia', 'population': 37833000},
    'beijing_china': {'lat': 39.9042, 'lng': 116.4074, 'region': 'Asia', 'population': 21540000},
    'shanghai_china': {'lat': 31.2304, 'lng': 121.4737, 'region': 'Asia', 'population': 27058000},
    'mumbai_india': {'lat': 19.0760, 'lng': 72.8777, 'region': 'Asia', 'population': 20411000},
    'delhi_india': {'lat': 28.7041, 'lng': 77.1025, 'region': 'Asia', 'population': 31870000},
    'singapore': {'lat': 1.3521, 'lng': 103.8198, 'region': 'Asia', 'population': 5896686},
    'seoul_south_korea': {'lat': 37.5665, 'lng': 126.9780, 'region': 'Asia', 'population': 9720846},
    'bangkok_thailand': {'lat': 13.7563, 'lng': 100.5018, 'region': 'Asia', 'population': 10719418},
    
    # Africa
    'cairo_egypt': {'lat': 30.0444, 'lng': 31.2357, 'region': 'Africa', 'population': 20900604},
    'lagos_nigeria': {'lat': 6.5244, 'lng': 3.3792, 'region': 'Africa', 'population': 14862000},
    'johannesburg_south_africa': {'lat': -26.2041, 'lng': 28.0473, 'region': 'Africa', 'population': 5635127},
    'casablanca_morocco': {'lat': 33.5731, 'lng': -7.5898, 'region': 'Africa', 'population': 3359818},
    
    # South America
    'sao_paulo_brazil': {'lat': -23.5505, 'lng': -46.6333, 'region': 'South America', 'population': 12325232},
    'buenos_aires_argentina': {'lat': -34.6118, 'lng': -58.3960, 'region': 'South America', 'population': 15594428},
    'lima_peru': {'lat': -12.0464, 'lng': -77.0428, 'region': 'South America', 'population': 10719418},
    'bogota_colombia': {'lat': 4.7110, 'lng': -74.0721, 'region': 'South America', 'population': 7181469},
    
    # Oceania
    'sydney_australia': {'lat': -33.8688, 'lng': 151.2093, 'region': 'Oceania', 'population': 5312163},
    'melbourne_australia': {'lat': -37.8136, 'lng': 144.9631, 'region': 'Oceania', 'population': 5078193},
    'auckland_new_zealand': {'lat': -36.8485, 'lng': 174.7633, 'region': 'Oceania', 'population': 1695200}
}

# ===============================
# GLOBAL STATE MANAGEMENT
# ===============================

live_processing_jobs = {}
live_training_sessions = {}
world_performance_metrics = {}
connected_clients = []
global_testing_results = {
    'total_locations_tested': 0,
    'successful_detections': 0,
    'average_global_iou': 0.0,
    'regional_performance': {},
    'real_time_metrics': []
}

# Performance tracking per region
for region in ['North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania']:
    global_testing_results['regional_performance'][region] = {
        'locations_tested': 0,
        'average_iou': 0.0,
        'best_technique': 'adaptive_fusion_rl',
        'processing_speed': 0.0
    }

# ===============================
# PYDANTIC MODELS
# ===============================

class LiveProcessingRequest(BaseModel):
    location: str
    technique: str = 'adaptive_fusion_rl'
    real_time_training: bool = True
    num_iterations: int = 10
    use_satellite_imagery: bool = True

class WorldMapTestingRequest(BaseModel):
    regions: List[str]
    num_locations_per_region: int = 5
    parallel_processing: bool = True
    training_enabled: bool = True
    performance_tracking: bool = True

class TrainingSessionRequest(BaseModel):
    session_name: str
    locations: List[str]
    training_episodes: int = 100
    batch_size: int = 64
    target_performance: float = 0.85

# ===============================
# SIMULATION FUNCTIONS
# ===============================

def generate_synthetic_satellite_image(lat: float, lng: float, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Generate synthetic satellite imagery based on coordinates."""
    height, width = size
    
    # Base terrain based on latitude (climate zones)
    if abs(lat) < 23.5:  # Tropical
        base_color = [34, 139, 34]  # Forest green
    elif abs(lat) < 35:  # Subtropical  
        base_color = [218, 165, 32]  # Goldenrod
    elif abs(lat) < 60:  # Temperate
        base_color = [107, 142, 35]  # Olive drab
    else:  # Polar
        base_color = [248, 248, 255]  # Ghost white
    
    # Create base image
    image = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # Add urban density based on population (from location data)
    location_key = None
    for key, data in WORLD_LOCATIONS.items():
        if abs(data['lat'] - lat) < 0.1 and abs(data['lng'] - lng) < 0.1:
            location_key = key
            break
    
    if location_key:
        population = WORLD_LOCATIONS[location_key]['population']
        urban_density = min(population / 1000000, 10) / 10  # Normalize to 0-1
        
        # Add buildings/urban structures
        num_buildings = int(50 * urban_density)
        for _ in range(num_buildings):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            bw = random.randint(10, 40)
            bh = random.randint(10, 40)
            
            # Building color (concrete/urban)
            building_color = [random.randint(100, 200) for _ in range(3)]
            cv2.rectangle(image, (x, y), (x + bw, y + bh), building_color, -1)
    
    # Add noise and texture
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def simulate_building_detection(image: np.ndarray, location: str) -> Dict[str, np.ndarray]:
    """Simulate building detection with multiple regularization methods."""
    height, width = image.shape[:2]
    
    # Simulate different regularization outputs
    outputs = {}
    
    # RT (Rectilinear Texture) - favors rectangular shapes
    rt_mask = np.zeros((height, width), dtype=np.float32)
    num_rect_buildings = random.randint(5, 15)
    for _ in range(num_rect_buildings):
        x = random.randint(0, width - 60)
        y = random.randint(0, height - 60)
        w = random.randint(20, 60)
        h = random.randint(20, 60)
        rt_mask[y:y+h, x:x+w] = random.uniform(0.7, 1.0)
    
    # RR (Rectangular Regularization) - clean rectangular shapes
    rr_mask = np.zeros((height, width), dtype=np.float32)
    num_clean_buildings = random.randint(3, 10)
    for _ in range(num_clean_buildings):
        x = random.randint(0, width - 50)
        y = random.randint(0, height - 50)
        w = random.randint(25, 50)
        h = random.randint(25, 50)
        rr_mask[y:y+h, x:x+w] = random.uniform(0.8, 1.0)
    
    # FER (Fast Edge Regularization) - edge-preserving
    fer_mask = np.zeros((height, width), dtype=np.float32)
    num_edge_buildings = random.randint(4, 12)
    for _ in range(num_edge_buildings):
        center_x = random.randint(30, width - 30)
        center_y = random.randint(30, height - 30)
        radius = random.randint(15, 35)
        cv2.circle(fer_mask, (center_x, center_y), radius, random.uniform(0.6, 0.9), -1)
    
    # Add realistic noise
    for mask in [rt_mask, rr_mask, fer_mask]:
        noise = np.random.normal(0, 0.05, mask.shape)
        mask += noise
        np.clip(mask, 0, 1, out=mask)
    
    outputs['rt'] = rt_mask
    outputs['rr'] = rr_mask  
    outputs['fer'] = fer_mask
    
    return outputs

async def process_location_with_rl(location: str, 
                                 training: bool = True, 
                                 num_iterations: int = 10) -> Dict:
    """Process a single location with RL adaptive fusion."""
    
    if location not in WORLD_LOCATIONS:
        raise ValueError(f"Unknown location: {location}")
    
    location_data = WORLD_LOCATIONS[location]
    lat, lng = location_data['lat'], location_data['lng']
    
    # Generate synthetic satellite imagery
    satellite_image = generate_synthetic_satellite_image(lat, lng)
    
    # Simulate building detection
    reg_outputs = simulate_building_detection(satellite_image, location)
    
    # Generate synthetic ground truth
    ground_truth = np.zeros_like(reg_outputs['rt'])
    num_gt_buildings = random.randint(3, 8)
    for _ in range(num_gt_buildings):
        x = random.randint(0, ground_truth.shape[1] - 40)
        y = random.randint(0, ground_truth.shape[0] - 40) 
        w = random.randint(20, 40)
        h = random.randint(20, 40)
        ground_truth[y:y+h, x:x+w] = 1.0
    
    results = {
        'location': location,
        'coordinates': {'lat': lat, 'lng': lng},
        'region': location_data['region'],
        'iterations': [],
        'final_performance': {},
        'rl_stats': {}
    }
    
    best_iou = 0.0
    best_weights = None
    
    for iteration in range(num_iterations):
        # Extract features
        state = rl_agent.extract_enhanced_features(
            reg_outputs, 
            satellite_image, 
            (lat, lng)
        )
        
        # Select action
        action = rl_agent.select_action(state, training=training)
        weights = rl_agent.action_to_weights[action]
        
        # Fuse masks
        fused_mask = rl_agent.fuse_masks(reg_outputs, action)
        
        # Compute reward
        reward = rl_agent.compute_reward(fused_mask, ground_truth)
        iou, precision, recall, f1 = rl_agent._compute_metrics(fused_mask, ground_truth)
        
        # Store experience if training
        if training:
            # Use same state as next state for simplicity
            rl_agent.memory.push(state, action, reward, state, reward > 0.9)
            
            # Train if enough experience
            loss = rl_agent.train_step()
            
            # Update target network periodically  
            if iteration % 10 == 0:
                rl_agent.update_target_network()
        
        # Track best performance
        if iou > best_iou:
            best_iou = iou
            best_weights = weights
        
        # Record iteration results
        results['iterations'].append({
            'iteration': iteration,
            'action': action,
            'weights': {
                'rt': weights[0],
                'rr': weights[1], 
                'fer': weights[2],
                'mask_rcnn': weights[3]
            },
            'metrics': {
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'reward': reward
            }
        })
        
        # Small delay for realistic processing
        await asyncio.sleep(0.01)
    
    # Final results
    results['final_performance'] = {
        'best_iou': best_iou,
        'best_weights': best_weights,
        'convergence_iteration': len([x for x in results['iterations'] if x['metrics']['iou'] > best_iou * 0.9])
    }
    
    results['rl_stats'] = {
        'epsilon': rl_agent.epsilon,
        'memory_size': len(rl_agent.memory),
        'total_episodes': rl_agent.global_metrics['total_training_episodes']
    }
    
    # Update global metrics
    rl_agent.global_metrics['total_processed'] += 1
    if training:
        rl_agent.global_metrics['total_training_episodes'] += num_iterations
    
    # Update running averages
    current_avg = rl_agent.global_metrics['average_iou']
    n = rl_agent.global_metrics['total_processed']
    rl_agent.global_metrics['average_iou'] = ((current_avg * (n-1)) + best_iou) / n
    rl_agent.global_metrics['best_iou'] = max(rl_agent.global_metrics['best_iou'], best_iou)
    
    return results

# ===============================
# API ENDPOINTS  
# ===============================

@app.get('/')
async def root():
    return {
        'message': 'GeoAI Adaptive Fusion with RL Agent - Live World Map Testing',
        'version': '3.0.0',
        'status': 'operational',
        'device': str(device),
        'rl_agent_status': 'active',
        'world_locations': len(WORLD_LOCATIONS),
        'active_jobs': len(live_processing_jobs)
    }

@app.get('/api/world-locations')
async def get_world_locations():
    """Get all available world locations for testing."""
    return {
        'locations': WORLD_LOCATIONS,
        'total_count': len(WORLD_LOCATIONS),
        'regions': list(set([loc['region'] for loc in WORLD_LOCATIONS.values()]))
    }

@app.get('/api/rl-agent/status')
async def get_rl_agent_status():
    """Get current RL agent status and performance."""
    return {
        'agent_status': 'active',
        'device': str(device),
        'network_parameters': {
            'state_dim': rl_agent.state_dim,
            'action_dim': rl_agent.action_dim,
            'epsilon': rl_agent.epsilon,
            'memory_size': len(rl_agent.memory),
        },
        'performance_metrics': rl_agent.global_metrics,
        'training_status': {
            'episodes_completed': rl_agent.global_metrics['total_training_episodes'],
            'convergence_rate': rl_agent.global_metrics.get('convergence_rate', 0.0),
            'adaptation_speed': rl_agent.global_metrics.get('adaptation_speed', 0.0)
        }
    }

@app.post('/api/process/single-location')
async def process_single_location(request: LiveProcessingRequest, background_tasks: BackgroundTasks):
    """Process a single location with RL adaptive fusion."""
    
    if request.location not in WORLD_LOCATIONS:
        raise HTTPException(status_code=400, detail=f"Unknown location: {request.location}")
    
    job_id = f"live_rl_{int(datetime.now().timestamp())}_{request.location}"
    
    live_processing_jobs[job_id] = {
        'job_id': job_id,
        'location': request.location,
        'technique': request.technique,
        'status': 'starting',
        'progress': 0,
        'start_time': datetime.now().isoformat(),
        'real_time_training': request.real_time_training,
        'num_iterations': request.num_iterations
    }
    
    # Start background processing
    background_tasks.add_task(
        process_single_location_bg,
        job_id, 
        request.location,
        request.real_time_training,
        request.num_iterations
    )
    
    return {
        'job_id': job_id,
        'status': 'started',
        'location': request.location,
        'estimated_completion': f"{request.num_iterations * 0.5} seconds"
    }

async def process_single_location_bg(job_id: str, 
                                   location: str,
                                   training: bool,
                                   num_iterations: int):
    """Background task for single location processing."""
    
    try:
        live_processing_jobs[job_id]['status'] = 'processing'
        live_processing_jobs[job_id]['progress'] = 10
        
        # Process with RL
        results = await process_location_with_rl(location, training, num_iterations)
        
        live_processing_jobs[job_id]['status'] = 'completed'
        live_processing_jobs[job_id]['progress'] = 100
        live_processing_jobs[job_id]['end_time'] = datetime.now().isoformat()
        live_processing_jobs[job_id]['results'] = results
        
        # Update global testing results
        global_testing_results['total_locations_tested'] += 1
        if results['final_performance']['best_iou'] > 0.5:
            global_testing_results['successful_detections'] += 1
        
        # Update regional performance
        region = results['region']
        if region in global_testing_results['regional_performance']:
            region_stats = global_testing_results['regional_performance'][region]
            region_stats['locations_tested'] += 1
            
            # Update running average
            n = region_stats['locations_tested']
            old_avg = region_stats['average_iou']
            new_iou = results['final_performance']['best_iou']
            region_stats['average_iou'] = ((old_avg * (n-1)) + new_iou) / n
        
        logger.info(f"Completed processing for location {location} with IoU: {results['final_performance']['best_iou']:.3f}")
        
    except Exception as e:
        live_processing_jobs[job_id]['status'] = 'error'
        live_processing_jobs[job_id]['error'] = str(e)
        logger.error(f"Error processing location {location}: {str(e)}")

@app.get('/api/process/status/{job_id}')
async def get_processing_status(job_id: str):
    """Get status of a processing job."""
    
    if job_id not in live_processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return live_processing_jobs[job_id]

@app.post('/api/world-map/batch-test')
async def start_world_map_batch_test(request: WorldMapTestingRequest, background_tasks: BackgroundTasks):
    """Start batch testing across multiple world locations."""
    
    session_id = f"world_test_{int(datetime.now().timestamp())}"
    
    # Select locations for testing
    selected_locations = []
    for region in request.regions:
        region_locations = [
            name for name, data in WORLD_LOCATIONS.items() 
            if data['region'] == region
        ]
        
        # Randomly select locations from this region
        num_to_select = min(request.num_locations_per_region, len(region_locations))
        selected_locations.extend(random.sample(region_locations, num_to_select))
    
    live_training_sessions[session_id] = {
        'session_id': session_id,
        'status': 'starting',
        'progress': 0,
        'start_time': datetime.now().isoformat(),
        'selected_locations': selected_locations,
        'total_locations': len(selected_locations),
        'completed_locations': 0,
        'results': [],
        'parallel_processing': request.parallel_processing,
        'training_enabled': request.training_enabled
    }
    
    # Start background batch processing
    background_tasks.add_task(
        process_world_map_batch_bg,
        session_id,
        selected_locations, 
        request.parallel_processing,
        request.training_enabled
    )
    
    return {
        'session_id': session_id,
        'status': 'started', 
        'selected_locations': selected_locations,
        'total_locations': len(selected_locations),
        'estimated_duration': f"{len(selected_locations) * 5} seconds"
    }

async def process_world_map_batch_bg(session_id: str,
                                   locations: List[str],
                                   parallel_processing: bool,
                                   training_enabled: bool):
    """Background task for batch world map testing."""
    
    try:
        live_training_sessions[session_id]['status'] = 'processing'
        live_training_sessions[session_id]['progress'] = 5
        
        results = []
        
        if parallel_processing:
            # Process locations in parallel
            tasks = []
            for location in locations:
                task = process_location_with_rl(location, training_enabled, 20)
                tasks.append(task)
            
            # Wait for all tasks to complete
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing location {locations[i]}: {result}")
                    continue
                
                results.append(result)
                live_training_sessions[session_id]['completed_locations'] = len(results)
                live_training_sessions[session_id]['progress'] = int((len(results) / len(locations)) * 90) + 5
                
        else:
            # Process locations sequentially
            for i, location in enumerate(locations):
                try:
                    result = await process_location_with_rl(location, training_enabled, 20)
                    results.append(result)
                    
                    live_training_sessions[session_id]['completed_locations'] = len(results)
                    live_training_sessions[session_id]['progress'] = int(((i + 1) / len(locations)) * 90) + 5
                    
                except Exception as e:
                    logger.error(f"Error processing location {location}: {e}")
                    continue
        
        # Compute summary statistics
        successful_results = [r for r in results if r['final_performance']['best_iou'] > 0.3]
        
        summary = {
            'total_processed': len(results),
            'successful_detections': len(successful_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'average_iou': np.mean([r['final_performance']['best_iou'] for r in results]) if results else 0,
            'regional_breakdown': {}
        }
        
        # Regional breakdown
        for result in results:
            region = result['region']
            if region not in summary['regional_breakdown']:
                summary['regional_breakdown'][region] = {
                    'count': 0,
                    'average_iou': 0.0,
                    'best_iou': 0.0
                }
            
            summary['regional_breakdown'][region]['count'] += 1
            region_ious = [r['final_performance']['best_iou'] for r in results if r['region'] == region]
            summary['regional_breakdown'][region]['average_iou'] = np.mean(region_ious)
            summary['regional_breakdown'][region]['best_iou'] = np.max(region_ious)
        
        live_training_sessions[session_id]['status'] = 'completed'
        live_training_sessions[session_id]['progress'] = 100
        live_training_sessions[session_id]['end_time'] = datetime.now().isoformat()
        live_training_sessions[session_id]['results'] = results
        live_training_sessions[session_id]['summary'] = summary
        
        logger.info(f"Completed batch testing for session {session_id}: {len(results)} locations processed")
        
    except Exception as e:
        live_training_sessions[session_id]['status'] = 'error'
        live_training_sessions[session_id]['error'] = str(e)
        logger.error(f"Error in batch processing session {session_id}: {str(e)}")

@app.get('/api/world-map/session/{session_id}')
async def get_batch_session_status(session_id: str):
    """Get status of a batch testing session."""
    
    if session_id not in live_training_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return live_training_sessions[session_id]

@app.get('/api/world-map/global-metrics')
async def get_global_testing_metrics():
    """Get global testing performance metrics."""
    
    # Update global average
    total = global_testing_results['total_locations_tested']
    if total > 0:
        success_rate = global_testing_results['successful_detections'] / total
        global_testing_results['global_success_rate'] = success_rate
    
    return {
        **global_testing_results,
        'rl_agent_metrics': rl_agent.global_metrics,
        'timestamp': datetime.now().isoformat()
    }

@app.websocket("/ws/live-metrics")
async def websocket_live_metrics(websocket: WebSocket):
    """WebSocket endpoint for live performance metrics."""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            # Send live metrics every 2 seconds
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'rl_agent': rl_agent.global_metrics,
                'global_testing': global_testing_results,
                'active_jobs': len(live_processing_jobs),
                'active_sessions': len(live_training_sessions)
            }
            
            await websocket.send_json(metrics)
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)