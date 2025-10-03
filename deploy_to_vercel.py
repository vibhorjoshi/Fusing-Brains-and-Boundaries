#!/usr/bin/env python
"""
Vercel Deployment Script for GeoAI

This script facilitates the deployment of the GeoAI project to Vercel.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Constants
VERCEL_CONFIG = "vercel.json"
FRONTEND_DIR = "frontend"
API_DIR = "api"
REQUIRED_ENV_VARS = ["VERCEL_TOKEN", "VERCEL_ORG_ID", "VERCEL_PROJECT_ID"]

def check_prerequisites():
    """Check that all prerequisites are met"""
    # Check if Vercel CLI is installed
    try:
        subprocess.run(["vercel", "--version"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Vercel CLI is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Vercel CLI is not installed. Please install it with: npm i -g vercel")
        return False
    
    # Check for required environment variables
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them before deploying.")
        return False
    
    # Check if vercel.json exists
    if not os.path.exists(VERCEL_CONFIG):
        print(f"❌ {VERCEL_CONFIG} not found. Please create it first.")
        return False
    
    # Check if frontend package.json exists
    if not os.path.exists(os.path.join(FRONTEND_DIR, "package.json")):
        print(f"❌ {FRONTEND_DIR}/package.json not found.")
        return False
    
    print("✅ All prerequisites checked")
    return True

def prepare_api_for_vercel():
    """Prepare the API for Vercel deployment"""
    print("\nPreparing API for Vercel deployment...")
    
    # Create API directory if it doesn't exist
    os.makedirs(API_DIR, exist_ok=True)
    
    # Create requirements.txt for the API
    with open(os.path.join(API_DIR, "requirements.txt"), "w") as f:
        f.write("fastapi==0.104.0\n")
        f.write("uvicorn==0.24.0\n")
        f.write("pydantic==2.4.0\n")
        f.write("numpy==1.24.0\n")
        f.write("pandas==2.0.0\n")
        f.write("pillow==10.0.0\n")
        f.write("python-multipart==0.0.6\n")
        
    # Create a simple API file if it doesn't exist
    api_file = os.path.join(API_DIR, "index.py")
    if not os.path.exists(api_file):
        with open(api_file, "w") as f:
            f.write("""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/api/regions")
async def get_regions():
    """Get available regions"""
    return {
        "regions": ["Alabama", "Mississippi", "Georgia", "Florida", "Tennessee"]
    }
""")
    print("✅ API prepared for Vercel")

def deploy_to_vercel():
    """Deploy the project to Vercel"""
    print("\nDeploying to Vercel...")
    
    # Run Vercel deployment
    try:
        result = subprocess.run(
            ["vercel", "--prod"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        print("✅ Deployment to Vercel completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e.stderr}")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("GeoAI Vercel Deployment")
    print("=" * 50)
    
    if not check_prerequisites():
        sys.exit(1)
    
    prepare_api_for_vercel()
    
    if deploy_to_vercel():
        print("\n" + "=" * 50)
        print("🎉 GeoAI is now deployed on Vercel!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ Deployment failed. Please check the logs above.")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()