"""
Simple Demo Server for GeoAI
This is a minimal FastAPI server that provides a health endpoint and basic API functionality
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="GeoAI Demo API",
    description="Simple demonstration API for GeoAI project",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Server is running"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GeoAI Demo API",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/api/demo")
async def demo_endpoint():
    """Demo endpoint that returns sample data"""
    return {
        "demo": True,
        "timestamp": "2023-06-15T12:34:56Z",
        "data": {
            "sample": "This is sample data from the GeoAI Demo API"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 8002))
    host = os.environ.get("API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)