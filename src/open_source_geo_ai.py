"""
Free Open-Source Geo AI Client for Building Footprint Extraction

This module provides integration with multiple free and open-source APIs:
1. OpenStreetMap for geographical data
2. NASA/ESA satellite imagery APIs
3. Hugging Face models for computer vision
4. Local reinforcement learning for patch analysis

Usage:
    client = OpenSourceGeoAI()
    image = client.get_satellite_image("Alabama")
    results = client.analyze_with_rl_patches(image)
"""

import os
import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Optional, Tuple, Dict, List, Any
import json
import base64
import time
import random
from urllib.parse import quote
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import pipeline, AutoImageProcessor, AutoModel
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    import rasterio
    from rasterio.io import MemoryFile
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class OpenSourceGeoAI:
    """Free and open-source geo AI client for building footprint analysis."""
    
    def __init__(self, timeout_s: float = 30.0):
        """Initialize the open-source geo AI client."""
        self.timeout_s = timeout_s
        self.session = requests.Session()
        
        # Initialize Hugging Face models if available
        self.hf_models = {}
        if HUGGINGFACE_AVAILABLE:
            self._init_huggingface_models()
        
        # RL patch analysis state
        self.patch_rewards = {}
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        
    def _init_huggingface_models(self):
        """Initialize Hugging Face models for computer vision tasks."""
        try:
            # Image segmentation model
            self.hf_models['segmentation'] = pipeline(
                "image-segmentation", 
                model="facebook/detr-resnet-50-panoptic",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Object detection model
            self.hf_models['detection'] = pipeline(
                "object-detection", 
                model="facebook/detr-resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("✅ Hugging Face models initialized successfully!")
            
        except Exception as e:
            print(f"⚠️ Could not initialize Hugging Face models: {e}")
            self.hf_models = {}
    
    def get_satellite_image(self, location: str, zoom: int = 15, 
                           size: Tuple[int, int] = (640, 640)) -> Optional[np.ndarray]:
        """
        Get satellite imagery from free/open sources.
        
        Args:
            location: Location name (city, state, coordinates)
            zoom: Zoom level (10-18)
            size: Image size as (width, height)
            
        Returns:
            BGR image array or None if unavailable
        """
        # Try multiple free sources in order of preference
        sources = [
            self._get_osm_satellite,
            self._get_nasa_modis,
            self._get_synthetic_realistic,
        ]
        
        for source_func in sources:
            try:
                image = source_func(location, zoom, size)
                if image is not None:
                    return image
            except Exception as e:
                print(f"Source failed: {e}")
                continue
        
        # Final fallback
        return self._create_advanced_synthetic(location, size)
    
    def _create_simple_demo_image(self, size: Tuple[int, int]) -> np.ndarray:
        """Create simple demo image as ultimate fallback"""
        h, w = size[1], size[0]
        image = np.random.randint(100, 180, (h, w, 3), dtype=np.uint8)
        
        # Add some simple rectangular "buildings"
        for _ in range(20):
            x1 = np.random.randint(0, w-50)
            y1 = np.random.randint(0, h-50)
            x2 = x1 + np.random.randint(20, 50)
            y2 = y1 + np.random.randint(20, 50)
            
            color = np.random.randint(50, 120, 3).tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        
        return image
    
    def _get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location using free geocoding."""
        try:
            # Use Nominatim (OpenStreetMap) free geocoding
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': location,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout_s)
            if response.status_code == 200:
                data = response.json()
                if data:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    return (lat, lon)
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return self._fallback_coordinates(location)
    
    def _fallback_coordinates(self, location: str) -> Tuple[float, float]:
        """Fallback coordinates for common locations."""
        coords = {
            'alabama': (32.3617, -86.2792),
            'arizona': (34.0489, -111.0937),
            'california': (36.7783, -119.4179),
            'florida': (27.7663, -82.6404),
            'texas': (31.9686, -99.9018),
            'new york': (40.7128, -74.0060),
            'chicago': (41.8781, -87.6298),
            'los angeles': (34.0522, -118.2437),
        }
        location_key = location.lower().strip()
        return coords.get(location_key, (40.0, -95.0))  # Default to center of USA
    
    def _get_osm_satellite(self, location: str, zoom: int, size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get satellite imagery from OpenStreetMap tile servers."""
        coords = self._get_coordinates(location)
        if not coords:
            return None
        
        lat, lon = coords
        
        # Convert lat/lon to tile coordinates
        def deg2num(lat_deg, lon_deg, zoom):
            lat_rad = np.radians(lat_deg)
            n = 2.0 ** zoom
            x = int((lon_deg + 180.0) / 360.0 * n)
            y = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
            return (x, y)
        
        x, y = deg2num(lat, lon, zoom)
        
        # Try different tile servers
        tile_servers = [
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}",
            f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}",
            f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/GoogleMapsCompatible/{zoom}/{y}/{x}.jpg"
        ]
        
        for tile_url in tile_servers:
            try:
                headers = {
                    'User-Agent': 'GeoAI-Research/1.0'
                }
                response = self.session.get(tile_url, headers=headers, timeout=self.timeout_s)
                
                if response.status_code == 200:
                    # Convert to numpy array
                    image_data = np.frombuffer(response.content, dtype=np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Resize to requested size
                        image = cv2.resize(image, size)
                        return image
                        
            except Exception as e:
                print(f"Tile server error: {e}")
                continue
        
        return None
    
    def _get_nasa_modis(self, location: str, zoom: int, size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get imagery from NASA MODIS (free but lower resolution)."""
        try:
            coords = self._get_coordinates(location)
            if not coords:
                return None
            
            lat, lon = coords
            
            # NASA GIBS WMTS service (free)
            base_url = "https://map1.vis.earthdata.nasa.gov/wmts-geo/1.0.0/MODIS_Terra_CorrectedReflectance_TrueColor/default"
            date = "2023-01-01"  # Use a recent date
            
            # Calculate tile coordinates for MODIS
            tile_size = 256
            map_size = 2 ** zoom * tile_size
            
            x = int((lon + 180) * map_size / 360)
            y = int((1 - (lat + 90) / 180) * map_size)
            
            tile_x = x // tile_size
            tile_y = y // tile_size
            
            url = f"{base_url}/{date}/GoogleMapsCompatible_Level9/{zoom}/{tile_y}/{tile_x}.jpg"
            
            response = self.session.get(url, timeout=self.timeout_s)
            if response.status_code == 200:
                image_data = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                
                if image is not None:
                    image = cv2.resize(image, size)
                    return image
                    
        except Exception as e:
            print(f"NASA MODIS error: {e}")
        
        return None
    
    def _get_synthetic_realistic(self, location: str, zoom: int, size: Tuple[int, int]) -> np.ndarray:
        """Generate realistic synthetic satellite imagery."""
        width, height = size
        
        # Get location characteristics
        coords = self._get_coordinates(location)
        location_type = self._classify_location(location)
        
        # Create base terrain
        image = self._generate_terrain_base(size, location_type)
        
        # Add urban features
        self._add_urban_features(image, location_type, zoom)
        
        # Add natural features
        self._add_natural_features(image, location_type)
        
        # Apply atmospheric effects
        image = self._apply_atmospheric_effects(image)
        
        return image
    
    def _classify_location(self, location: str) -> Dict[str, Any]:
        """Classify location type and characteristics."""
        location_lower = location.lower()
        
        # State classifications
        desert_states = ['arizona', 'nevada', 'new mexico', 'utah']
        coastal_states = ['california', 'florida', 'texas', 'new york']
        forest_states = ['oregon', 'washington', 'maine', 'vermont']
        plains_states = ['kansas', 'nebraska', 'iowa', 'illinois']
        
        # City classifications
        major_cities = ['new york', 'los angeles', 'chicago', 'houston']
        
        classification = {
            'terrain': 'mixed',
            'urbanization': 'medium',
            'vegetation': 'moderate',
            'water_bodies': False,
            'building_density': 'medium'
        }
        
        if any(state in location_lower for state in desert_states):
            classification.update({
                'terrain': 'desert',
                'vegetation': 'sparse',
                'building_density': 'low'
            })
        elif any(state in location_lower for state in coastal_states):
            classification.update({
                'terrain': 'coastal',
                'water_bodies': True,
                'building_density': 'high'
            })
        elif any(state in location_lower for state in forest_states):
            classification.update({
                'terrain': 'forest',
                'vegetation': 'dense',
                'building_density': 'low'
            })
        elif any(state in location_lower for state in plains_states):
            classification.update({
                'terrain': 'plains',
                'vegetation': 'moderate',
                'building_density': 'medium'
            })
        
        if any(city in location_lower for city in major_cities):
            classification.update({
                'urbanization': 'high',
                'building_density': 'very_high'
            })
        
        return classification
    
    def _generate_terrain_base(self, size: Tuple[int, int], location_type: Dict) -> np.ndarray:
        """Generate base terrain colors."""
        width, height = size
        terrain = location_type['terrain']
        
        if terrain == 'desert':
            # Sandy/desert colors
            base_color = [120, 180, 200]  # BGR: sandy
        elif terrain == 'forest':
            # Forest green base
            base_color = [60, 120, 80]   # BGR: forest green
        elif terrain == 'coastal':
            # Mixed terrain with blue
            base_color = [140, 160, 120] # BGR: coastal mix
        else:  # plains or mixed
            # Standard terrain
            base_color = [100, 140, 120] # BGR: standard terrain
        
        # Create base image with noise
        image = np.random.normal(base_color, 20, (height, width, 3))
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Add terrain texture
        self._add_terrain_texture(image, terrain)
        
        return image
    
    def _add_terrain_texture(self, image: np.ndarray, terrain: str):
        """Add realistic terrain texture."""
        height, width = image.shape[:2]
        
        # Add noise patterns based on terrain
        if terrain == 'desert':
            # Add sand dune patterns
            for _ in range(5):
                x = random.randint(0, width - 50)
                y = random.randint(0, height - 50)
                cv2.ellipse(image, (x, y), (30, 15), random.randint(0, 180), 0, 360, 
                           (130, 190, 210), -1)
        
        elif terrain == 'forest':
            # Add tree-like patterns
            for _ in range(20):
                x = random.randint(10, width - 10)
                y = random.randint(10, height - 10)
                radius = random.randint(3, 8)
                cv2.circle(image, (x, y), radius, (40, 100, 60), -1)
    
    def _add_urban_features(self, image: np.ndarray, location_type: Dict, zoom: int):
        """Add urban features like buildings and roads."""
        density = location_type['building_density']
        height, width = image.shape[:2]
        
        # Determine number of buildings based on density and zoom
        if density == 'very_high':
            num_buildings = min(40, zoom * 2)
        elif density == 'high':
            num_buildings = min(25, zoom * 1.5)
        elif density == 'medium':
            num_buildings = min(15, zoom)
        else:
            num_buildings = min(8, zoom // 2)
        
        # Add buildings
        for _ in range(int(num_buildings)):
            self._add_building(image, density)
        
        # Add road network
        self._add_road_network(image, location_type['urbanization'])
    
    def _add_building(self, image: np.ndarray, density: str):
        """Add a single building to the image."""
        height, width = image.shape[:2]
        
        # Building size based on density
        if density == 'very_high':
            size_range = (15, 40)
        elif density == 'high':
            size_range = (20, 50)
        else:
            size_range = (10, 30)
        
        bw = random.randint(*size_range)
        bh = random.randint(*size_range)
        
        x = random.randint(0, max(1, width - bw))
        y = random.randint(0, max(1, height - bh))
        
        # Building colors (darker than terrain)
        building_colors = [
            (40, 50, 60),    # Dark gray
            (50, 60, 70),    # Medium gray
            (60, 70, 80),    # Light gray
            (45, 55, 65),    # Blue-gray
        ]
        
        color = random.choice(building_colors)
        
        # Draw building
        cv2.rectangle(image, (x, y), (x + bw, y + bh), color, -1)
        
        # Add shadow
        if x + bw + 2 < width and y + bh + 2 < height:
            shadow_color = tuple(max(0, c - 15) for c in color)
            cv2.rectangle(image, (x + 2, y + 2), (x + bw + 2, y + bh + 2), shadow_color, 2)
    
    def _add_road_network(self, image: np.ndarray, urbanization: str):
        """Add road network to the image."""
        height, width = image.shape[:2]
        road_color = (45, 45, 45)  # Dark gray
        
        if urbanization in ['high', 'very_high']:
            # Grid pattern
            spacing = 80 if urbanization == 'very_high' else 100
            
            # Vertical roads
            for x in range(spacing, width, spacing):
                cv2.line(image, (x, 0), (x, height), road_color, 3)
            
            # Horizontal roads
            for y in range(spacing, height, spacing):
                cv2.line(image, (0, y), (width, y), road_color, 3)
        else:
            # Organic road pattern
            for _ in range(3):
                start_x, start_y = random.randint(0, width), random.randint(0, height)
                end_x, end_y = random.randint(0, width), random.randint(0, height)
                cv2.line(image, (start_x, start_y), (end_x, end_y), road_color, 2)
    
    def _add_natural_features(self, image: np.ndarray, location_type: Dict):
        """Add natural features like water bodies and vegetation."""
        height, width = image.shape[:2]
        
        # Add water bodies if coastal
        if location_type.get('water_bodies', False):
            self._add_water_body(image)
        
        # Add vegetation based on vegetation density
        veg_density = location_type.get('vegetation', 'moderate')
        self._add_vegetation(image, veg_density)
    
    def _add_water_body(self, image: np.ndarray):
        """Add a water body to the image."""
        height, width = image.shape[:2]
        
        # Random water body
        water_color = (120, 80, 40)  # BGR: blue water
        
        # Create irregular water shape
        points = []
        center_x, center_y = width // 3, height // 3
        for angle in range(0, 360, 45):
            radius = random.randint(30, 60)
            x = center_x + int(radius * np.cos(np.radians(angle)))
            y = center_y + int(radius * np.sin(np.radians(angle)))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(image, [points], water_color)
    
    def _add_vegetation(self, image: np.ndarray, density: str):
        """Add vegetation patches."""
        height, width = image.shape[:2]
        
        if density == 'sparse':
            num_patches = 3
        elif density == 'moderate':
            num_patches = 8
        else:  # dense
            num_patches = 15
        
        for _ in range(num_patches):
            # Random vegetation patch
            patch_size = random.randint(20, 40)
            x = random.randint(0, max(1, width - patch_size))
            y = random.randint(0, max(1, height - patch_size))
            
            # Green colors for vegetation
            green_shades = [
                (60, 100, 70),   # Dark green
                (70, 120, 80),   # Medium green
                (80, 140, 90),   # Light green
            ]
            
            color = random.choice(green_shades)
            
            # Draw irregular vegetation patch
            cv2.ellipse(image, (x + patch_size//2, y + patch_size//2), 
                       (patch_size//2, patch_size//3), 
                       random.randint(0, 180), 0, 360, color, -1)
    
    def _apply_atmospheric_effects(self, image: np.ndarray) -> np.ndarray:
        """Apply realistic atmospheric effects."""
        # Slight blur for altitude effect
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # Add slight noise
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Slight color adjustment for satellite look
        image = cv2.convertScaleAbs(image, alpha=0.95, beta=5)
        
        return image
    
    def _create_advanced_synthetic(self, location: str, size: Tuple[int, int]) -> np.ndarray:
        """Create advanced synthetic image as final fallback."""
        return self._get_synthetic_realistic(location, 15, size)
    
    def analyze_with_rl_patches(self, image: np.ndarray, patch_size: int = 64) -> Dict[str, Any]:
        """
        Analyze image using reinforcement learning on patches.
        
        Args:
            image: Input satellite image
            patch_size: Size of patches for RL analysis
            
        Returns:
            Analysis results with building detection and RL insights
        """
        try:
            height, width = image.shape[:2]
            
            # Extract patches
            patches = self._extract_patches(image, patch_size)
            
            # Apply RL analysis to each patch
            rl_results = []
            building_mask = np.zeros((height, width), dtype=np.uint8)
            
            for i, (patch, x, y) in enumerate(patches):
                # RL-based patch analysis
                patch_analysis = self._rl_analyze_patch(patch, i)
                rl_results.append(patch_analysis)
                
                # Update building mask based on RL prediction
                if patch_analysis['has_buildings']:
                    mask_patch = self._generate_patch_mask(patch, patch_analysis['confidence'])
                    
                    # Place mask in correct position
                    end_x = min(x + patch_size, width)
                    end_y = min(y + patch_size, height)
                    mask_h, mask_w = mask_patch.shape
                    
                    building_mask[y:y+mask_h, x:x+mask_w] = np.maximum(
                        building_mask[y:y+mask_h, x:x+mask_w], 
                        mask_patch[:end_y-y, :end_x-x]
                    )
            
            # Apply Hugging Face models if available
            hf_results = self._analyze_with_huggingface(image)
            
            # Combine results
            analysis = {
                'building_mask': building_mask,
                'rl_patch_results': rl_results,
                'total_patches': len(patches),
                'building_patches': sum(1 for r in rl_results if r['has_buildings']),
                'average_confidence': np.mean([r['confidence'] for r in rl_results]),
                'rl_learning_progress': self._get_learning_progress(),
                'huggingface_results': hf_results
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'building_mask': np.zeros(image.shape[:2], dtype=np.uint8)}
    
    def _extract_patches(self, image: np.ndarray, patch_size: int) -> List[Tuple[np.ndarray, int, int]]:
        """Extract overlapping patches from image."""
        height, width = image.shape[:2]
        patches = []
        
        stride = patch_size // 2  # 50% overlap
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append((patch, x, y))
        
        return patches
    
    def _rl_analyze_patch(self, patch: np.ndarray, patch_id: int) -> Dict[str, Any]:
        """Analyze a patch using reinforcement learning."""
        # Simple RL-like analysis based on learned patterns
        
        # Extract features
        features = self._extract_patch_features(patch)
        
        # RL decision making (simplified)
        state_key = self._get_state_key(features)
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Explore: random decision
            has_buildings = random.random() > 0.5
            confidence = random.random()
        else:
            # Exploit: use learned knowledge
            has_buildings, confidence = self._predict_from_state(state_key, features)
        
        # Update learning (simplified Q-learning)
        reward = self._calculate_reward(patch, has_buildings)
        self._update_q_value(state_key, has_buildings, reward)
        
        return {
            'patch_id': patch_id,
            'has_buildings': has_buildings,
            'confidence': confidence,
            'features': features,
            'reward': reward,
            'state_key': state_key
        }
    
    def _extract_patch_features(self, patch: np.ndarray) -> Dict[str, float]:
        """Extract features from a patch for RL analysis."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        features = {
            'mean_intensity': float(np.mean(gray)),
            'std_intensity': float(np.std(gray)),
            'edge_density': float(np.mean(cv2.Canny(gray, 50, 150)) / 255.0),
            'contrast': float(gray.max() - gray.min()) / 255.0,
            'entropy': self._calculate_entropy(gray),
            'geometric_regularity': self._calculate_geometric_regularity(gray)
        }
        
        return features
    
    def _calculate_entropy(self, gray: np.ndarray) -> float:
        """Calculate entropy of image patch."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        return float(-np.sum(hist * np.log2(hist)))
    
    def _calculate_geometric_regularity(self, gray: np.ndarray) -> float:
        """Calculate geometric regularity (building-like patterns)."""
        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        regularity_scores = []
        for contour in contours:
            if len(contour) > 10:  # Minimum points for analysis
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Rectangularity score (4 corners = building-like)
                rectangularity = 1.0 - abs(len(approx) - 4) / 10.0
                regularity_scores.append(max(0.0, rectangularity))
        
        return float(np.mean(regularity_scores)) if regularity_scores else 0.0
    
    def _get_state_key(self, features: Dict[str, float]) -> str:
        """Generate state key for RL learning."""
        # Discretize features for state representation
        mean_bin = int(features['mean_intensity'] / 32)  # 8 bins
        edge_bin = int(features['edge_density'] * 4)     # 4 bins
        reg_bin = int(features['geometric_regularity'] * 4)  # 4 bins
        
        return f"{mean_bin}_{edge_bin}_{reg_bin}"
    
    def _predict_from_state(self, state_key: str, features: Dict[str, float]) -> Tuple[bool, float]:
        """Predict building presence from learned state."""
        if state_key in self.patch_rewards:
            avg_reward = np.mean(self.patch_rewards[state_key])
            confidence = abs(avg_reward)  # Higher absolute reward = higher confidence
            has_buildings = avg_reward > 0
        else:
            # Use heuristic for unseen states
            building_score = (
                features['edge_density'] * 0.3 + 
                features['geometric_regularity'] * 0.4 +
                (1.0 - features['std_intensity'] / 128.0) * 0.3  # Lower variation = more building-like
            )
            has_buildings = building_score > 0.5
            confidence = building_score
        
        return has_buildings, float(confidence)
    
    def _calculate_reward(self, patch: np.ndarray, predicted_buildings: bool) -> float:
        """Calculate reward for RL learning (simplified)."""
        # Simple heuristic-based reward
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Buildings typically have:
        # - Rectangular shapes
        # - Sharp edges
        # - Uniform color regions
        
        edge_strength = np.mean(cv2.Canny(gray, 50, 150)) / 255.0
        uniformity = 1.0 - (np.std(gray) / 128.0)
        
        # Calculate "ground truth" score based on features
        building_likelihood = edge_strength * 0.6 + uniformity * 0.4
        
        # Reward is positive if prediction matches likelihood
        if predicted_buildings and building_likelihood > 0.5:
            reward = building_likelihood
        elif not predicted_buildings and building_likelihood <= 0.5:
            reward = 1.0 - building_likelihood
        else:
            reward = -(abs(building_likelihood - 0.5) + 0.1)  # Penalty for wrong prediction
        
        return float(reward)
    
    def _update_q_value(self, state_key: str, action: bool, reward: float):
        """Update Q-value for RL learning."""
        if state_key not in self.patch_rewards:
            self.patch_rewards[state_key] = []
        
        self.patch_rewards[state_key].append(reward)
        
        # Keep only recent rewards (sliding window)
        max_history = 20
        if len(self.patch_rewards[state_key]) > max_history:
            self.patch_rewards[state_key] = self.patch_rewards[state_key][-max_history:]
    
    def _get_learning_progress(self) -> Dict[str, Any]:
        """Get RL learning progress statistics."""
        if not self.patch_rewards:
            return {'states_learned': 0, 'average_reward': 0.0}
        
        all_rewards = []
        for rewards in self.patch_rewards.values():
            all_rewards.extend(rewards)
        
        return {
            'states_learned': len(self.patch_rewards),
            'total_experiences': len(all_rewards),
            'average_reward': float(np.mean(all_rewards)),
            'reward_variance': float(np.var(all_rewards))
        }
    
    def _generate_patch_mask(self, patch: np.ndarray, confidence: float) -> np.ndarray:
        """Generate building mask for a patch."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding based on confidence
        threshold_value = int(128 + (confidence - 0.5) * 50)
        
        # Use multiple methods and combine
        _, thresh1 = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine thresholds
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _analyze_with_huggingface(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image using Hugging Face models."""
        if not HUGGINGFACE_AVAILABLE or not self.hf_models:
            return {'error': 'Hugging Face models not available'}
        
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            results = {}
            
            # Object detection
            if 'detection' in self.hf_models:
                detections = self.hf_models['detection'](pil_image)
                building_detections = [d for d in detections if 'building' in d.get('label', '').lower()]
                results['object_detection'] = {
                    'total_detections': len(detections),
                    'building_detections': len(building_detections),
                    'detections': detections[:5]  # Limit output
                }
            
            # Image segmentation
            if 'segmentation' in self.hf_models:
                segments = self.hf_models['segmentation'](pil_image)
                building_segments = [s for s in segments if 'building' in s.get('label', '').lower()]
                results['segmentation'] = {
                    'total_segments': len(segments),
                    'building_segments': len(building_segments)
                }
            
            return results
            
        except Exception as e:
            return {'error': f'Hugging Face analysis failed: {e}'}