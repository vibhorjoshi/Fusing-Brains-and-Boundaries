"""
API Client for FastAPI Backend
Handles communication between frontend and backend
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class GeoAIAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_cities(self) -> List[Dict[str, Any]]:
        """Get all Alabama cities data"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/cities") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get cities: {response.status}")
    
    async def get_city(self, city_name: str) -> Dict[str, Any]:
        """Get specific city data"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/cities/{city_name}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get city {city_name}: {response.status}")
    
    async def get_3d_buildings(self, city_name: str, limit: int = 100) -> Dict[str, Any]:
        """Get 3D buildings for a city"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/3d/buildings/{city_name}?limit={limit}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get 3D buildings for {city_name}: {response.status}")
    
    async def start_processing(self, city_name: str, processing_type: str = "full_pipeline") -> Dict[str, Any]:
        """Start processing job for a city"""
        payload = {
            "city_name": city_name,
            "processing_type": processing_type,
            "options": {}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/process", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to start processing for {city_name}: {response.status}")
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/jobs/{job_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get job {job_id}: {response.status}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/metrics") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get metrics: {response.status}")
    
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/analytics/performance") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get performance analytics: {response.status}")

# Test the API client
async def test_api():
    client = GeoAIAPIClient()
    
    try:
        print("Testing API client...")
        
        # Test cities endpoint
        cities = await client.get_cities()
        print(f"✅ Cities: {len(cities)} found")
        
        # Test specific city
        if cities:
            city_name = cities[0]["name"]
            city_data = await client.get_city(city_name)
            print(f"✅ City data for {city_name}: {city_data['buildings']} buildings")
            
            # Test 3D buildings
            buildings_3d = await client.get_3d_buildings(city_name, limit=10)
            print(f"✅ 3D Buildings for {city_name}: {len(buildings_3d['buildings'])} samples")
            
            # Test processing
            job_response = await client.start_processing(city_name)
            print(f"✅ Processing job started: {job_response['job_id']}")
        
        # Test metrics
        metrics = await client.get_metrics()
        print(f"✅ System metrics: {metrics['total_buildings']} total buildings")
        
        # Test analytics
        analytics = await client.get_performance_analytics()
        print(f"✅ Performance analytics: {len(analytics['cities'])} cities analyzed")
        
        print("🎉 All API tests passed!")
        
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_api())