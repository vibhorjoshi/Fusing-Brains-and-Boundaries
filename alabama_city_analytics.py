# Alabama City Performance Analytics API
# This file provides backend API endpoints for the Alabama cities visualization

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import json
import os
from pathlib import Path

# Import unified configuration
try:
    from unified_config import Config
except ImportError:
    # Fallback config if main config is not available
    class Config:
        DATA_DIR = "data"
        ALABAMA_DATA_FILE = "alabama_cities_performance.json"

# Create router
router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["analytics"],
    responses={404: {"description": "Not found"}},
)

# Data models
class CityMetrics(BaseModel):
    name: str
    metrics: Dict[str, float]

class AlabamaCityPerformance(BaseModel):
    cities: List[CityMetrics]
    timestamp: str
    version: str

# Sample data - will be loaded from disk if available
DEFAULT_ALABAMA_DATA = {
    "cities": [
        {
            "name": "Birmingham",
            "metrics": {
                "iou": 73.4,
                "buildings_detected": 156421,
                "processing_time": 26.4,
                "precision": 78.2,
                "recall": 75.6
            }
        },
        {
            "name": "Montgomery",
            "metrics": {
                "iou": 71.2,
                "buildings_detected": 98742,
                "processing_time": 18.7,
                "precision": 76.8,
                "recall": 73.1
            }
        },
        {
            "name": "Mobile",
            "metrics": {
                "iou": 69.8,
                "buildings_detected": 87634,
                "processing_time": 17.2,
                "precision": 74.5,
                "recall": 71.9
            }
        },
        {
            "name": "Huntsville",
            "metrics": {
                "iou": 78.6,
                "buildings_detected": 124563,
                "processing_time": 22.9,
                "precision": 82.3,
                "recall": 79.4
            }
        },
        {
            "name": "Tuscaloosa",
            "metrics": {
                "iou": 72.1,
                "buildings_detected": 65432,
                "processing_time": 14.8,
                "precision": 76.3,
                "recall": 74.2
            }
        },
        {
            "name": "Auburn",
            "metrics": {
                "iou": 70.5,
                "buildings_detected": 42387,
                "processing_time": 10.5,
                "precision": 75.2,
                "recall": 72.4
            }
        },
        {
            "name": "Dothan",
            "metrics": {
                "iou": 67.2,
                "buildings_detected": 38921,
                "processing_time": 9.7,
                "precision": 72.8,
                "recall": 69.5
            }
        },
        {
            "name": "Hoover",
            "metrics": {
                "iou": 74.3,
                "buildings_detected": 47532,
                "processing_time": 12.3,
                "precision": 78.9,
                "recall": 76.1
            }
        },
        {
            "name": "Decatur",
            "metrics": {
                "iou": 69.1,
                "buildings_detected": 32156,
                "processing_time": 8.6,
                "precision": 73.4,
                "recall": 70.8
            }
        }
    ],
    "timestamp": "2023-05-15T08:30:00Z",
    "version": "1.0.0"
}

# Helper functions
def load_alabama_data():
    """Load Alabama cities performance data from disk or use default"""
    data_dir = Path(Config.DATA_DIR)
    data_file = data_dir / Config.ALABAMA_DATA_FILE
    
    if data_file.exists():
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading Alabama data: {e}")
    
    return DEFAULT_ALABAMA_DATA

def save_alabama_data(data):
    """Save Alabama cities performance data to disk"""
    data_dir = Path(Config.DATA_DIR)
    data_file = data_dir / Config.ALABAMA_DATA_FILE
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving Alabama data: {e}")
        return False

# API Endpoints
@router.get("/alabama_performance", response_model=AlabamaCityPerformance)
async def get_alabama_performance():
    """Get performance metrics for Alabama cities"""
    data = load_alabama_data()
    return data

@router.get("/alabama_city/{city_name}")
async def get_city_metrics(city_name: str):
    """Get detailed performance metrics for a specific Alabama city"""
    data = load_alabama_data()
    
    for city in data["cities"]:
        if city["name"].lower() == city_name.lower():
            return city
    
    raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

@router.post("/update_alabama_data")
async def update_alabama_data(data: AlabamaCityPerformance):
    """Update Alabama cities performance data"""
    if save_alabama_data(data.dict()):
        return {"status": "success", "message": "Data updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update data")

@router.get("/city_comparison")
async def compare_cities(
    cities: List[str] = Query(None, description="List of cities to compare"),
    metrics: List[str] = Query(None, description="List of metrics to compare")
):
    """Compare performance metrics across multiple Alabama cities"""
    data = load_alabama_data()
    
    if not cities:
        # Default to all cities
        cities = [city["name"] for city in data["cities"]]
    
    if not metrics:
        # Default to all metrics
        if data["cities"] and "metrics" in data["cities"][0]:
            metrics = list(data["cities"][0]["metrics"].keys())
        else:
            metrics = []
    
    comparison = {
        "cities": [],
        "metrics": metrics
    }
    
    for city_data in data["cities"]:
        if city_data["name"] in cities:
            city_metrics = {
                "name": city_data["name"],
                "values": {}
            }
            
            for metric in metrics:
                if metric in city_data["metrics"]:
                    city_metrics["values"][metric] = city_data["metrics"][metric]
            
            comparison["cities"].append(city_metrics)
    
    return comparison

# Register the router with the app in unified_backend.py
# from alabama_city_analytics import router
# app.include_router(router)