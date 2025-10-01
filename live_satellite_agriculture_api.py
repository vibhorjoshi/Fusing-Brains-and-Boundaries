from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import random
import time
import logging
import math
import base64
from io import BytesIO
import os

# Enhanced API Configuration for Live Satellite Processing
API_KEYS = {
    "satellite-agriculture-live-2024": "Live Satellite Agricultural Detection",
    "usa-agriculture-api-key-2024": "USA Agriculture Detection System",
    "nasa-satellite-key": "NASA Satellite Data Access",
    "admin-access-key": "Administrator Access"
}

SATELLITE_APIS = {
    "nasa": {
        "endpoint": "https://api.nasa.gov/planetary/earth/imagery",
        "key": os.getenv("NASA_API_KEY", "demo-nasa-key"),
        "resolution": "10m"
    },
    "google": {
        "endpoint": "https://earthengine.googleapis.com/v1/projects",
        "key": os.getenv("GOOGLE_EARTH_API_KEY", "demo-google-key"),
        "resolution": "10m"
    },
    "sentinel": {
        "endpoint": "https://scihub.copernicus.eu/dhus/search",
        "key": os.getenv("SENTINEL_API_KEY", "demo-sentinel-key"),
        "resolution": "10m"
    },
    "modis": {
        "endpoint": "https://modis.gsfc.nasa.gov/data/dataprod",
        "key": os.getenv("MODIS_API_KEY", "demo-modis-key"),
        "resolution": "250m"
    }
}

app = FastAPI(
    title='🛰️ Live Agricultural Detection - Satellite Integration API',
    version='5.0.0',
    description='Real-time agricultural and building detection using satellite imagery with adaptive fusion RL',
    contact={
        "name": "GeoAI Satellite Research Team",
        "email": "satellite@geoai.research"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Valid API key required")
    return api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# LIVE SATELLITE DATA MODELS
# ===============================

class SatelliteImageRequest(BaseModel):
    provider: str = "nasa"  # nasa, google, sentinel, modis
    lat: float
    lng: float
    zoom: int = 10
    resolution: str = "10m"
    date: Optional[str] = None
    cloud_cover_max: int = 30

class LiveDetectionRequest(BaseModel):
    satellite_provider: str = "nasa"
    region: str = "all"  # state name or "all"
    detection_types: List[str] = ["crop", "building"]
    adaptive_fusion: bool = True
    real_time_training: bool = True
    update_frequency: int = 5  # seconds

class RegionBounds(BaseModel):
    north: float
    south: float
    east: float
    west: float
    name: str

# ===============================
# USA AGRICULTURAL REGIONS DATA
# ===============================

USA_AGRICULTURAL_REGIONS = {
    "california": {
        "name": "California",
        "bounds": {"north": 42.0, "south": 32.5, "east": -114.1, "west": -124.4},
        "major_crops": ["almonds", "grapes", "strawberries", "lettuce", "tomatoes"],
        "agricultural_area": 25400000,  # acres
        "farms": 69900,
        "avg_farm_size": 312,
        "crop_seasons": {
            "spring": ["lettuce", "strawberries", "artichokes"],
            "summer": ["tomatoes", "grapes", "melons"],
            "fall": ["almonds", "walnuts", "citrus"]
        }
    },
    "texas": {
        "name": "Texas", 
        "bounds": {"north": 36.5, "south": 25.8, "east": -93.5, "west": -106.6},
        "major_crops": ["cotton", "corn", "wheat", "sorghum", "rice"],
        "agricultural_area": 127000000,
        "farms": 247000,
        "avg_farm_size": 514,
        "crop_seasons": {
            "spring": ["corn", "cotton", "rice"],
            "summer": ["sorghum", "soybeans"],
            "fall": ["wheat", "winter_wheat"]
        }
    },
    "iowa": {
        "name": "Iowa",
        "bounds": {"north": 43.5, "south": 40.4, "east": -90.1, "west": -96.6},
        "major_crops": ["corn", "soybeans"],
        "agricultural_area": 23000000,
        "farms": 86900,
        "avg_farm_size": 359,
        "crop_seasons": {
            "spring": ["corn", "soybeans"],
            "summer": ["corn", "soybeans"],
            "fall": ["harvest_corn", "harvest_soybeans"]
        }
    },
    "illinois": {
        "name": "Illinois",
        "bounds": {"north": 42.5, "south": 36.9, "east": -87.0, "west": -91.5},
        "major_crops": ["corn", "soybeans", "wheat"],
        "agricultural_area": 22000000,
        "farms": 72200,
        "avg_farm_size": 358,
        "crop_seasons": {
            "spring": ["corn", "soybeans"],
            "summer": ["corn", "soybeans"],
            "fall": ["wheat", "harvest"]
        }
    },
    "nebraska": {
        "name": "Nebraska", 
        "bounds": {"north": 43.0, "south": 40.0, "east": -95.3, "west": -104.0},
        "major_crops": ["corn", "soybeans", "wheat", "sorghum"],
        "agricultural_area": 19500000,
        "farms": 46350,
        "avg_farm_size": 946,
        "crop_seasons": {
            "spring": ["corn", "soybeans"],
            "summer": ["corn", "sorghum"],
            "fall": ["wheat", "harvest"]
        }
    }
}

# ===============================
# ADAPTIVE FUSION RL AGENT FOR SATELLITE IMAGERY
# ===============================

class SatelliteAdaptiveFusionRL:
    def __init__(self):
        self.state_size = 25  # Enhanced for satellite data
        self.action_size = 100  # More actions for satellite processing
        self.memory = []
        self.epsilon = 0.8  # Higher exploration for new satellite data
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.total_episodes = 0
        
        # Satellite-specific metrics
        self.satellite_metrics = {
            "images_processed": 0,
            "cloud_cover_filtered": 0,
            "detection_accuracy": 0.0,
            "processing_speed": 0.0,
            "data_latency": 0.0
        }
        
        # Performance tracking
        self.performance_history = []
        
    def get_satellite_state(self, image_data, region_info, weather_data):
        """Extract state features from satellite imagery and context"""
        return [
            image_data.get('cloud_cover', 0) / 100,
            image_data.get('resolution_quality', 1.0),
            region_info.get('agricultural_area', 0) / 100000000,
            region_info.get('farms', 0) / 100000,
            len(region_info.get('major_crops', [])) / 10,
            weather_data.get('temperature', 20) / 50,
            weather_data.get('precipitation', 0) / 100,
            weather_data.get('humidity', 50) / 100,
            # Season encoding
            datetime.now().month / 12,
            datetime.now().hour / 24,
            # Historical performance
            sum(self.performance_history[-10:]) / 10 if self.performance_history else 0.5,
            # Processing metrics
            self.satellite_metrics['detection_accuracy'],
            self.satellite_metrics['processing_speed'] / 10,
            self.satellite_metrics['data_latency'] / 10,
            # Additional satellite-specific features
            *[random.random() for _ in range(11)]  # Simulated satellite sensor data
        ]
    
    def choose_satellite_action(self, state):
        """Choose processing action for satellite image analysis"""
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Simulate Q-value calculation for satellite processing
        q_values = [random.random() for _ in range(self.action_size)]
        return q_values.index(max(q_values))
    
    def process_satellite_detection(self, image_data, region, crop_season):
        """Process satellite imagery for agricultural detection"""
        processing_start = time.time()
        
        # Extract features from satellite data
        state = self.get_satellite_state(
            image_data, 
            USA_AGRICULTURAL_REGIONS.get(region, {}),
            {"temperature": 25, "precipitation": 10, "humidity": 60}
        )
        
        # Choose processing action
        action = self.choose_satellite_action(state)
        
        # Simulate satellite image processing
        results = self.simulate_satellite_analysis(image_data, action, region, crop_season)
        
        # Calculate processing time
        processing_time = time.time() - processing_start
        self.satellite_metrics['processing_speed'] = processing_time
        
        # Update metrics
        self.satellite_metrics['images_processed'] += 1
        self.satellite_metrics['detection_accuracy'] = results['accuracy']
        
        # Store performance
        self.performance_history.append(results['accuracy'])
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.total_episodes += 1
        
        return results
    
    def simulate_satellite_analysis(self, image_data, action, region, season):
        """Simulate AI analysis of satellite imagery"""
        
        # Base accuracy varies by satellite provider
        provider_accuracy = {
            "nasa": 0.85,
            "google": 0.88,
            "sentinel": 0.82,
            "modis": 0.75
        }
        
        base_accuracy = provider_accuracy.get(image_data.get('provider', 'nasa'), 0.80)
        
        # Adjust for cloud cover
        cloud_penalty = image_data.get('cloud_cover', 0) * 0.01
        
        # Adjust for resolution
        resolution_bonus = (20 - float(image_data.get('resolution', '10').rstrip('m'))) * 0.01
        
        # Season adjustment for crop detection
        seasonal_crops = USA_AGRICULTURAL_REGIONS.get(region, {}).get('crop_seasons', {}).get(season, [])
        season_bonus = len(seasonal_crops) * 0.02
        
        # Calculate final accuracies
        crop_accuracy = max(0.5, min(0.95, 
            base_accuracy + resolution_bonus + season_bonus - cloud_penalty + random.uniform(-0.05, 0.05)
        ))
        
        building_accuracy = max(0.6, min(0.92,
            base_accuracy + resolution_bonus - cloud_penalty * 0.5 + random.uniform(-0.03, 0.03)
        ))
        
        # Simulate detection counts
        total_detections = random.randint(50, 200)
        crop_detections = int(total_detections * 0.7)
        building_detections = int(total_detections * 0.3)
        
        return {
            'accuracy': (crop_accuracy + building_accuracy) / 2,
            'crop_accuracy': crop_accuracy,
            'building_accuracy': building_accuracy,
            'detections': {
                'total': total_detections,
                'crops': crop_detections,
                'buildings': building_detections
            },
            'processing_time': self.satellite_metrics['processing_speed'],
            'confidence': random.uniform(0.8, 0.95),
            'detected_crops': random.sample(seasonal_crops, min(3, len(seasonal_crops))) if seasonal_crops else []
        }

# Global RL Agent Instance
satellite_rl_agent = SatelliteAdaptiveFusionRL()

# ===============================
# LIVE PROCESSING MANAGEMENT
# ===============================

active_live_sessions = {}
websocket_connections = []

# ===============================
# API ENDPOINTS
# ===============================

@app.get("/")
async def root():
    return {
        "message": "🛰️ Live Agricultural Detection - Satellite Integration API",
        "version": "5.0.0",
        "status": "operational",
        "satellite_providers": list(SATELLITE_APIS.keys()),
        "supported_regions": list(USA_AGRICULTURAL_REGIONS.keys()),
        "active_sessions": len(active_live_sessions),
        "rl_agent_status": "active",
        "features": [
            "Real-time satellite imagery processing",
            "Multi-provider satellite data integration",
            "Adaptive fusion RL for agricultural detection", 
            "Live performance monitoring",
            "USA regional crop season analysis"
        ]
    }

@app.get("/api/satellite/providers")
async def get_satellite_providers():
    """Get available satellite data providers"""
    return {
        "providers": SATELLITE_APIS,
        "recommended": "nasa",
        "real_time_capable": ["nasa", "google", "sentinel"],
        "historical_data": ["modis", "sentinel"]
    }

@app.get("/api/usa/agricultural-regions")
async def get_agricultural_regions():
    """Get USA agricultural regions data"""
    return {
        "regions": USA_AGRICULTURAL_REGIONS,
        "total_regions": len(USA_AGRICULTURAL_REGIONS),
        "total_agricultural_area": sum(r["agricultural_area"] for r in USA_AGRICULTURAL_REGIONS.values()),
        "total_farms": sum(r["farms"] for r in USA_AGRICULTURAL_REGIONS.values())
    }

@app.post("/api/satellite/image")
async def get_satellite_image(request: SatelliteImageRequest, api_key: str = Depends(verify_api_key)):
    """Fetch satellite imagery for specified coordinates"""
    
    provider_config = SATELLITE_APIS.get(request.provider)
    if not provider_config:
        raise HTTPException(status_code=400, detail=f"Unknown satellite provider: {request.provider}")
    
    # Simulate satellite image fetching
    image_data = {
        "provider": request.provider,
        "coordinates": {"lat": request.lat, "lng": request.lng},
        "zoom": request.zoom,
        "resolution": request.resolution,
        "timestamp": datetime.now().isoformat(),
        "cloud_cover": random.randint(5, 40),
        "quality_score": random.uniform(0.8, 0.98),
        "image_url": f"https://satellite-api.example.com/{request.provider}/image_{int(time.time())}.jpg",
        "metadata": {
            "satellite": f"{request.provider.upper()}-{random.randint(1, 3)}",
            "resolution_actual": f"{random.randint(8, 12)}m",
            "acquisition_time": datetime.now().isoformat(),
            "sun_elevation": random.randint(30, 70),
            "sensor_angle": random.randint(-15, 15)
        }
    }
    
    return {
        "success": True,
        "image_data": image_data,
        "api_usage": {
            "key_name": API_KEYS.get(api_key),
            "requests_remaining": random.randint(800, 1000)
        }
    }

@app.post("/api/live/start-detection")
async def start_live_detection(request: LiveDetectionRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    """Start live agricultural detection using satellite imagery"""
    
    session_id = f"live_{int(datetime.now().timestamp())}"
    
    # Initialize live session
    active_live_sessions[session_id] = {
        "session_id": session_id,
        "status": "starting",
        "provider": request.satellite_provider,
        "region": request.region,
        "detection_types": request.detection_types,
        "adaptive_fusion": request.adaptive_fusion,
        "real_time_training": request.real_time_training,
        "update_frequency": request.update_frequency,
        "start_time": datetime.now().isoformat(),
        "images_processed": 0,
        "detections_count": 0,
        "current_accuracy": 0.0
    }
    
    # Start background processing
    background_tasks.add_task(process_live_satellite_detection, session_id)
    
    return {
        "success": True,
        "session_id": session_id,
        "message": f"Live detection started for {request.region} using {request.satellite_provider} satellite",
        "estimated_first_results": f"{request.update_frequency + 2}s"
    }

async def process_live_satellite_detection(session_id: str):
    """Background task for continuous satellite image processing"""
    
    session = active_live_sessions.get(session_id)
    if not session:
        return
    
    session["status"] = "processing"
    
    try:
        while session_id in active_live_sessions and session["status"] == "processing":
            
            # Get region info
            region_info = USA_AGRICULTURAL_REGIONS.get(session["region"], USA_AGRICULTURAL_REGIONS["california"])
            bounds = region_info["bounds"]
            
            # Generate random coordinates within region
            lat = random.uniform(bounds["south"], bounds["north"])
            lng = random.uniform(bounds["west"], bounds["east"])
            
            # Simulate satellite image data
            image_data = {
                "provider": session["provider"],
                "coordinates": {"lat": lat, "lng": lng},
                "cloud_cover": random.randint(5, 30),
                "resolution": "10m",
                "quality_score": random.uniform(0.85, 0.98)
            }
            
            # Process with RL agent
            results = satellite_rl_agent.process_satellite_detection(
                image_data, 
                session["region"], 
                "current"
            )
            
            # Update session
            session["images_processed"] += 1
            session["detections_count"] += results["detections"]["total"]
            session["current_accuracy"] = results["accuracy"]
            session["last_update"] = datetime.now().isoformat()
            
            # Broadcast to WebSocket connections
            await broadcast_live_update({
                "session_id": session_id,
                "detection_event": {
                    "coordinates": {"lat": lat, "lng": lng},
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                },
                "session_stats": {
                    "images_processed": session["images_processed"],
                    "total_detections": session["detections_count"],
                    "current_accuracy": session["current_accuracy"],
                    "rl_metrics": {
                        "epsilon": satellite_rl_agent.epsilon,
                        "total_episodes": satellite_rl_agent.total_episodes,
                        "avg_performance": sum(satellite_rl_agent.performance_history[-10:]) / 10 if satellite_rl_agent.performance_history else 0
                    }
                }
            })
            
            # Wait for next update
            await asyncio.sleep(session["update_frequency"])
            
    except Exception as e:
        logger.error(f"Error in live detection session {session_id}: {str(e)}")
        session["status"] = "error"
        session["error"] = str(e)

@app.get("/api/live/session/{session_id}")
async def get_live_session(session_id: str):
    """Get live detection session status"""
    
    session = active_live_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session": session,
        "rl_agent_stats": {
            "epsilon": satellite_rl_agent.epsilon,
            "total_episodes": satellite_rl_agent.total_episodes,
            "memory_size": len(satellite_rl_agent.memory),
            "avg_performance": sum(satellite_rl_agent.performance_history[-20:]) / 20 if satellite_rl_agent.performance_history else 0,
            "satellite_metrics": satellite_rl_agent.satellite_metrics
        }
    }

@app.delete("/api/live/session/{session_id}")
async def stop_live_session(session_id: str, api_key: str = Depends(verify_api_key)):
    """Stop live detection session"""
    
    if session_id not in active_live_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_live_sessions[session_id]
    session["status"] = "stopped"
    session["end_time"] = datetime.now().isoformat()
    
    # Remove from active sessions after a delay
    await asyncio.sleep(2)
    if session_id in active_live_sessions:
        del active_live_sessions[session_id]
    
    return {
        "success": True,
        "message": f"Live detection session {session_id} stopped",
        "final_stats": session
    }

@app.websocket("/ws/live-satellite-feed")
async def websocket_live_feed(websocket: WebSocket):
    """WebSocket endpoint for live satellite detection feed"""
    
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

async def broadcast_live_update(data):
    """Broadcast updates to all connected WebSocket clients"""
    
    if websocket_connections:
        disconnected = []
        
        for websocket in websocket_connections:
            try:
                await websocket.send_json(data)
            except:
                disconnected.append(websocket)
        
        # Clean up disconnected WebSocket connections
        for ws in disconnected:
            websocket_connections.remove(ws)

@app.get("/api/rl-agent/stats")
async def get_rl_agent_stats():
    """Get current RL agent statistics"""
    
    return {
        "agent_type": "Satellite Adaptive Fusion RL",
        "state_size": satellite_rl_agent.state_size,
        "action_size": satellite_rl_agent.action_size,
        "epsilon": satellite_rl_agent.epsilon,
        "total_episodes": satellite_rl_agent.total_episodes,
        "memory_size": len(satellite_rl_agent.memory),
        "satellite_metrics": satellite_rl_agent.satellite_metrics,
        "performance_trend": satellite_rl_agent.performance_history[-10:] if satellite_rl_agent.performance_history else [],
        "learning_stats": {
            "learning_rate": satellite_rl_agent.learning_rate,
            "epsilon_decay": satellite_rl_agent.epsilon_decay,
            "exploration_ratio": satellite_rl_agent.epsilon
        }
    }

if __name__ == "__main__":
    print("🛰️ Starting Live Agricultural Detection API with Satellite Integration...")
    uvicorn.run(app, host='0.0.0.0', port=8004)