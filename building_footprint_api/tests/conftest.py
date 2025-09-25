"""
Pytest configuration file
"""

import os
import pytest
from fastapi.testclient import TestClient
import asyncio
from typing import AsyncGenerator, Generator

import sys
import pathlib

# Add the parent directory to the path so we can import the app
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from main import app
from app.core.config import get_settings
from app.utils.db_utils import DatabaseManager

# Override settings for testing
os.environ["ENVIRONMENT"] = "testing"
os.environ["MONGODB_NAME"] = "building_footprint_test"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"  # Use DB 1 for testing

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client() -> Generator:
    """
    Create a test client for testing the API
    """
    with TestClient(app) as client:
        yield client

@pytest.fixture
async def db_manager() -> AsyncGenerator:
    """
    Create a database manager for testing
    """
    manager = DatabaseManager()
    await manager.initialize()
    
    # Clear test database collections
    await manager.db.jobs.delete_many({})
    await manager.db.results.delete_many({})
    
    yield manager
    
    # Cleanup
    await manager.db.jobs.delete_many({})
    await manager.db.results.delete_many({})
    await manager.close()

@pytest.fixture
def settings():
    """
    Get application settings
    """
    return get_settings()