from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import random
import time
import logging
from collections import deque
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

# ===============================
# SIMPLIFIED RL AGENT (No NumPy/PyTorch)
# ===============================

class SimplifiedAdaptiveFusionRL:
    """Simplified RL-based adaptive fusion without heavy dependencies."""
    
    def __init__(self):
        self.state_dim = 15
        self.action_dim = 64
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Simplified Q-table (dictionary-based)
        self.q_table = {}
        self.experience_replay = []
        self.max_replay_size = 10000
        
        # Performance tracking
        self.global_metrics = {
            'total_processed': 0,
            'total_training_episodes': 0,
            'average_iou': 0.0,
            'best_iou': 0.0,
            'convergence_rate': 0.0,
            'adaptation_speed': 0.0,
            'epsilon': self.epsilon,
            'memory_size': 0
        }
        
        # Action mapping for weight combinations
        self.action_to_weights = self._create_action_mapping()
        
    def _create_action_mapping(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Create action mapping with 4-component weights."""
        mapping = {}
        idx = 0
        
        # Generate weight combinations
        steps = 4  # 0.0, 0.33, 0.67, 1.0
        for w_rt in [i/(steps-1) for i in range(steps)]:
            for w_rr in [i/(steps-1) for i in range(steps)]:
                for w_fer in [i/(steps-1) for i in range(steps)]:
                    for w_mask in [i/(steps-1) for i in range(steps)]:
                        if idx < self.action_dim:
                            total = w_rt + w_rr + w_fer + w_mask
                            if total > 0:
                                mapping[idx] = (w_rt/total, w_rr/total, w_fer/total, w_mask/total)
                            else:
                                mapping[idx] = (0.25, 0.25, 0.25, 0.25)
                            idx += 1
        
        return mapping
    
    def extract_features(self, reg_outputs: Dict[str, List], location_info: Dict) -> List[float]:
        """Extract simplified features from regularization outputs."""
        features = []
        
        # Basic geometric features for each method
        for method in ["rt", "rr", "fer"]:
            if method in reg_outputs:
                mask_data = reg_outputs[method]
                # Simulate feature extraction
                area = sum(mask_data) / len(mask_data) if mask_data else 0
                perimeter = len([x for x in mask_data if x > 0.5])
                compactness = area / (perimeter + 1e-6) if perimeter > 0 else 0
                rectangularity = area * 0.8 + random.uniform(-0.1, 0.1)
                
                features.extend([area, perimeter/100, compactness, rectangularity])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Location-based features
        lat = location_info.get('lat', 0)
        lng = location_info.get('lng', 0)
        population = location_info.get('population', 0)
        
        features.extend([
            abs(lat) / 90.0,  # Normalized latitude
            (lng + 180) / 360.0,  # Normalized longitude  
            min(population / 10000000, 1.0)  # Normalized population
        ])
        
        return features[:self.state_dim]
    
    def select_action(self, state: List[float], training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        state_key = tuple([round(x, 3) for x in state])
        
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Get Q-values for state
        if state_key in self.q_table:
            q_values = self.q_table[state_key]
            return max(q_values, key=q_values.get)
        else:
            # Initialize new state
            self.q_table[state_key] = {i: random.uniform(-0.1, 0.1) for i in range(self.action_dim)}
            return random.randint(0, self.action_dim - 1)
    
    def compute_reward(self, action: int, location_info: Dict) -> float:
        """Compute reward based on action and location characteristics."""
        # Get weights from action
        weights = self.action_to_weights[action]
        
        # Simulate IoU based on location characteristics and weight balance
        population_factor = min(location_info.get('population', 0) / 1000000, 5) / 5
        weight_balance = 1.0 - abs(sum(weights) - 1.0)
        weight_variance = 1.0 / (1.0 + sum([(w - 0.25)**2 for w in weights]))
        
        base_iou = 0.3 + 0.4 * population_factor + 0.2 * weight_balance + 0.1 * weight_variance
        noise = random.uniform(-0.1, 0.1)
        
        iou = max(0.0, min(1.0, base_iou + noise))
        
        # Additional rewards for good performance
        reward = iou
        if iou > 0.8:
            reward += 0.1
        if iou > 0.9:
            reward += 0.05
            
        return reward, iou
    
    def update_q_table(self, state: List[float], action: int, reward: float, 
                      next_state: List[float], learning_rate: float = 0.1, 
                      discount_factor: float = 0.99):
        """Update Q-table using simplified Q-learning."""
        state_key = tuple([round(x, 3) for x in state])
        next_state_key = tuple([round(x, 3) for x in next_state])
        
        # Initialize states if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(self.action_dim)}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {i: 0.0 for i in range(self.action_dim)}
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())
        
        new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Store experience
        self.experience_replay.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })
        
        # Limit replay buffer
        if len(self.experience_replay) > self.max_replay_size:
            self.experience_replay.pop(0)
            
        self.global_metrics['memory_size'] = len(self.experience_replay)
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.global_metrics['epsilon'] = self.epsilon

# Global RL agent
rl_agent = SimplifiedAdaptiveFusionRL()

# ===============================
# WORLD LOCATIONS DATA
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
connected_clients = []
global_testing_results = {
    'total_locations_tested': 0,
    'successful_detections': 0,
    'average_global_iou': 0.0,
    'regional_performance': {},
    'real_time_metrics': []
}

# Initialize regional performance
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

# ===============================
# SIMULATION FUNCTIONS
# ===============================

def simulate_building_detection(location_info: Dict) -> Dict[str, List]:
    """Simulate building detection with multiple regularization methods."""
    
    # Generate synthetic regularization outputs based on location characteristics
    population = location_info['population']
    lat = location_info['lat']
    
    # Urban density affects detection quality
    urban_density = min(population / 1000000, 10) / 10
    climate_factor = 1.0 - abs(lat) / 90.0  # Temperate climates = better imagery
    
    base_quality = 0.3 + 0.4 * urban_density + 0.3 * climate_factor
    
    outputs = {}
    
    # RT (Rectilinear Texture) - rectangular shapes
    rt_values = []
    for _ in range(100):  # 10x10 grid simulation
        if random.random() < base_quality:
            rt_values.append(random.uniform(0.6, 1.0))
        else:
            rt_values.append(random.uniform(0.0, 0.4))
    outputs['rt'] = rt_values
    
    # RR (Rectangular Regularization) - clean rectangles
    rr_values = []
    for _ in range(100):
        if random.random() < base_quality * 0.8:  # Slightly more conservative
            rr_values.append(random.uniform(0.7, 1.0))
        else:
            rr_values.append(random.uniform(0.0, 0.3))
    outputs['rr'] = rr_values
    
    # FER (Fast Edge Regularization) - edge preservation
    fer_values = []
    for _ in range(100):
        if random.random() < base_quality * 1.1:  # Slightly more aggressive
            fer_values.append(random.uniform(0.5, 0.9))
        else:
            fer_values.append(random.uniform(0.0, 0.5))
    outputs['fer'] = fer_values
    
    return outputs

async def process_location_with_rl(location: str, 
                                 training: bool = True, 
                                 num_iterations: int = 10) -> Dict:
    """Process a single location with RL adaptive fusion."""
    
    if location not in WORLD_LOCATIONS:
        raise ValueError(f"Unknown location: {location}")
    
    location_data = WORLD_LOCATIONS[location]
    
    # Simulate building detection
    reg_outputs = simulate_building_detection(location_data)
    
    results = {
        'location': location,
        'coordinates': {'lat': location_data['lat'], 'lng': location_data['lng']},
        'region': location_data['region'],
        'iterations': [],
        'final_performance': {},
        'rl_stats': {}
    }
    
    best_iou = 0.0
    best_weights = None
    total_rewards = 0.0
    
    for iteration in range(num_iterations):
        # Extract features
        state = rl_agent.extract_features(reg_outputs, location_data)
        
        # Select action
        action = rl_agent.select_action(state, training=training)
        weights = rl_agent.action_to_weights[action]
        
        # Compute reward and IoU
        reward, iou = rl_agent.compute_reward(action, location_data)
        total_rewards += reward
        
        # Update RL agent if training
        if training:
            # Use same state as next state for simplicity
            next_state = state.copy()
            rl_agent.update_q_table(state, action, reward, next_state)
            rl_agent.decay_epsilon()
        
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
                'precision': iou * 0.9 + random.uniform(-0.05, 0.05),
                'recall': iou * 0.95 + random.uniform(-0.05, 0.05),
                'f1': iou * 0.92 + random.uniform(-0.05, 0.05),
                'reward': reward
            }
        })
        
        # Small delay for realistic processing
        await asyncio.sleep(0.01)
    
    # Final results
    convergence_iteration = len([x for x in results['iterations'] if x['metrics']['iou'] > best_iou * 0.9])
    
    results['final_performance'] = {
        'best_iou': best_iou,
        'best_weights': {
            'rt': best_weights[0] if best_weights else 0.25,
            'rr': best_weights[1] if best_weights else 0.25,
            'fer': best_weights[2] if best_weights else 0.25,
            'mask_rcnn': best_weights[3] if best_weights else 0.25
        },
        'average_reward': total_rewards / num_iterations,
        'convergence_iteration': convergence_iteration
    }
    
    results['rl_stats'] = {
        'epsilon': rl_agent.epsilon,
        'memory_size': len(rl_agent.experience_replay),
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
        'network_parameters': {
            'state_dim': rl_agent.state_dim,
            'action_dim': rl_agent.action_dim,
            'epsilon': rl_agent.epsilon,
            'memory_size': len(rl_agent.experience_replay),
            'q_table_size': len(rl_agent.q_table)
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
            'average_iou': sum([r['final_performance']['best_iou'] for r in results]) / len(results) if results else 0,
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
            summary['regional_breakdown'][region]['average_iou'] = sum(region_ious) / len(region_ious)
            summary['regional_breakdown'][region]['best_iou'] = max(region_ious)
        
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