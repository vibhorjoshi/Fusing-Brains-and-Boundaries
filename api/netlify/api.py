from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

# Initialize FastAPI app
app = FastAPI()

# Define models
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    environment: str

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "environment": "netlify"
    }

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)}
    )

# Netlify function handler
def handler(event, context):
    # Parse request
    path = event.get('path', '')
    http_method = event.get('httpMethod', '')
    
    # Return API information
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
        },
        'body': json.dumps({
            "status": "healthy",
            "service": "geoai-api-netlify-function",
            "version": "1.0.0",
            "path": path,
            "method": http_method,
            "timestamp": datetime.now().isoformat(),
            "message": "GeoAI API is running on Netlify Functions with limited functionality"
        })
    }