"""
Test health endpoints
"""

import pytest
from fastapi.testclient import TestClient

def test_health_check(test_client: TestClient):
    """Test basic health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data

def test_detailed_health_check(test_client: TestClient):
    """Test detailed health check endpoint"""
    response = test_client.get("/health/detailed")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "system" in data
    assert "cuda" in data
    assert "models" in data