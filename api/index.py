from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os
import sys
from datetime import datetime

# Gracefully handle missing optional dependencies
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available", file=sys.stderr)
    
try:
    import torch
except ImportError:
    print("Warning: PyTorch not available", file=sys.stderr)
    
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available", file=sys.stderr)

app = FastAPI(
    title="GeoAI API",
    description="API for the GeoAI Agricultural Detection System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

class RegionResponse(BaseModel):
    regions: List[str]

class AnalysisResult(BaseModel):
    region: str
    precision: float
    recall: float
    f1_score: float
    processing_time: float
    coverage: float
    samples: int

# Routes
@app.get("/api/health", response_model=HealthResponse)
async def health():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/regions", response_model=RegionResponse)
async def get_regions():
    """Get available regions for analysis"""
    return {
        "regions": ["Alabama", "Mississippi", "Georgia", "Florida", "Tennessee"]
    }

@app.get("/api/metrics/{region}", response_model=AnalysisResult)
async def get_region_metrics(region: str):
    """Get metrics for a specific region"""
    # In a real implementation, this would fetch from a database
    metrics = {
        "alabama": {
            "region": "Alabama",
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.905,
            "processing_time": 1.8,
            "coverage": 86.5,
            "samples": 7820
        },
        "mississippi": {
            "region": "Mississippi",
            "precision": 0.88,
            "recall": 0.91,
            "f1_score": 0.895,
            "processing_time": 2.1,
            "coverage": 82.3,
            "samples": 6950
        },
        "georgia": {
            "region": "Georgia",
            "precision": 0.94,
            "recall": 0.87,
            "f1_score": 0.904,
            "processing_time": 1.9,
            "coverage": 88.7,
            "samples": 8340
        },
        "florida": {
            "region": "Florida",
            "precision": 0.89,
            "recall": 0.86,
            "f1_score": 0.875,
            "processing_time": 2.3,
            "coverage": 76.4,
            "samples": 5780
        },
        "tennessee": {
            "region": "Tennessee",
            "precision": 0.91,
            "recall": 0.88,
            "f1_score": 0.895,
            "processing_time": 1.7,
            "coverage": 84.2,
            "samples": 7120
        }
    }
    
    region_lower = region.lower()
    if region_lower not in metrics:
        raise HTTPException(status_code=404, detail=f"Region {region} not found")
    
    return metrics[region_lower]

# Main entry point for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)