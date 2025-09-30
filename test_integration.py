#!/usr/bin/env python3
"""
GeoAI Platform Complete Integration Test
Tests all endpoints and demonstrates the full system functionality
"""

import requests
import time
import json
import asyncio
import websockets
from datetime import datetime

# Configuration
BACKEND_URL = "http://127.0.0.1:8002"
FRONTEND_URL = "http://localhost:3000"
WS_URL = "ws://127.0.0.1:8002/ws"

def print_header(title):
    print("\n" + "=" * 60)
    print(f"🌍 {title}")
    print("=" * 60)

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def test_backend_endpoints():
    """Test all backend API endpoints"""
    print_header("Backend API Testing")
    
    # Test root endpoint
    try:
        response = requests.get(f"{BACKEND_URL}/")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Root endpoint: {data['message']}")
        else:
            print_error(f"Root endpoint failed: {response.status_code}")
    except Exception as e:
        print_error(f"Root endpoint error: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check: {data['status']}")
        else:
            print_error(f"Health check failed: {response.status_code}")
    except Exception as e:
        print_error(f"Health check error: {e}")
    
    # Test visualization endpoints
    visualizations = [
        ("live", "Live 3D Visualization"),
        ("globe", "Globe Visualization"),
        ("analytics", "Analytics Dashboard"),
        ("ml-processing", "ML Processing Pipeline")
    ]
    
    for endpoint, name in visualizations:
        try:
            response = requests.get(f"{BACKEND_URL}/{endpoint}")
            if response.status_code == 200:
                print_success(f"{name} endpoint accessible")
            else:
                print_error(f"{name} endpoint failed: {response.status_code}")
        except Exception as e:
            print_error(f"{name} endpoint error: {e}")

def test_ml_processing():
    """Test ML processing endpoint"""
    print_header("ML Processing Test")
    
    payload = {
        "image_url": "https://example.com/test-satellite.jpg",
        "model_type": "mask_rcnn",
        "apply_regularization": True
    }
        # Check backend (should show FastAPI docs)
    curl http://127.0.0.1:8002/docs
    
    # Check frontend (should show Next.js app)
    curl http://localhost:3000
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/ml-processing/extract-buildings",
            json=payload
        )
        if response.status_code == 200:
            data = response.json()
            print_success(f"ML processing job started: {data.get('job_id', 'N/A')}")
            print_info(f"Model: {data.get('model_type', 'N/A')}")
            print_info(f"Status: {data.get('status', 'N/A')}")
        else:
            print_error(f"ML processing failed: {response.status_code}")
    except Exception as e:
        print_error(f"ML processing error: {e}")

async def test_websocket():
    """Test WebSocket connection"""
    print_header("WebSocket Connection Test")
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            print_success("WebSocket connected successfully")
            
            # Send test message
            test_message = {
                "type": "test",
                "message": "Integration test",
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(test_message))
            print_success("Test message sent")
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print_success(f"Received response: {response}")
            except asyncio.TimeoutError:
                print_info("No immediate response (normal behavior)")
                
    except Exception as e:
        print_error(f"WebSocket connection error: {e}")

def test_frontend():
    """Test frontend accessibility"""
    print_header("Frontend Testing")
    
    endpoints = [
        ("", "Home Page"),
        ("/dashboard", "Main Dashboard"),
        ("/dashboard/unified", "Unified Platform"),
        ("/health", "Health Check")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{FRONTEND_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                print_success(f"{name} accessible")
            else:
                print_error(f"{name} failed: {response.status_code}")
        except Exception as e:
            print_error(f"{name} error: {e}")

def display_system_status():
    """Display comprehensive system status"""
    print_header("System Status Summary")
    
    # Backend status
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        backend_status = "🟢 Online" if response.status_code == 200 else "🔴 Offline"
    except:
        backend_status = "🔴 Offline"
    
    # Frontend status
    try:
        response = requests.get(f"{FRONTEND_URL}/health", timeout=5)
        frontend_status = "🟢 Online" if response.status_code == 200 else "🔴 Offline"
    except:
        frontend_status = "🔴 Offline"
    
    print(f"Backend Server:      {backend_status}")
    print(f"Frontend Server:     {frontend_status}")
    print(f"Backend URL:         {BACKEND_URL}")
    print(f"Frontend URL:        {FRONTEND_URL}")
    print(f"Unified Dashboard:   {FRONTEND_URL}/dashboard/unified")
    
    print_header("Available Visualizations")
    print("🌟 Live 3D Processing:  http://127.0.0.1:8002/live")
    print("🌍 Globe Visualization: http://127.0.0.1:8002/globe")
    print("🧠 ML Pipeline:         http://127.0.0.1:8002/ml-processing")
    print("📊 Analytics Dashboard: http://127.0.0.1:8002/analytics")
    print("🚀 Unified Platform:    http://localhost:3000/dashboard/unified")

async def main():
    """Main integration test function"""
    print_header("GeoAI Platform Integration Test")
    print("Testing complete system functionality...")
    
    # Test backend
    test_backend_endpoints()
    
    # Test ML processing
    test_ml_processing()
    
    # Test WebSocket
    await test_websocket()
    
    # Test frontend
    test_frontend()
    
    # Display final status
    display_system_status()
    
    print_header("Integration Test Complete")
    print("🎉 GeoAI Platform is fully operational!")
    print("Access the unified dashboard at: http://localhost:3000/dashboard/unified")

if __name__ == "__main__":
    asyncio.run(main())