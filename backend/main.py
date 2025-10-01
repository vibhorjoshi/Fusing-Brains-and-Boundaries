from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
import json
from datetime import datetime

app = FastAPI(title='GeoAI 3D API', description='Advanced 3D Building Detection')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

cities_data = [
    {'name': 'Birmingham', 'lat': 33.5207, 'lng': -86.8025, 'buildings': 156421, 'accuracy': 91.2},
    {'name': 'Montgomery', 'lat': 32.3792, 'lng': -86.3077, 'buildings': 98742, 'accuracy': 89.7},
    {'name': 'Mobile', 'lat': 30.6954, 'lng': -88.0399, 'buildings': 87634, 'accuracy': 88.4},
    {'name': 'Huntsville', 'lat': 34.7304, 'lng': -86.5861, 'buildings': 124563, 'accuracy': 92.3},
    {'name': 'Tuscaloosa', 'lat': 33.2098, 'lng': -87.5692, 'buildings': 65432, 'accuracy': 90.1}
]

@app.get('/')
async def root():
    return {'message': 'GeoAI 3D Building Detection API', 'version': '3.0.0', 'status': 'operational'}

@app.get('/api/cities')
async def get_cities():
    return cities_data

@app.get('/api/cities/{city_name}')
async def get_city(city_name: str):
    city = next((c for c in cities_data if c['name'].lower() == city_name.lower()), None)
    if not city:
        raise HTTPException(status_code=404, detail='City not found')
    return city

@app.get('/api/3d/buildings/{city_name}')
async def get_3d_buildings(city_name: str, limit: int = 50):
    city = next((c for c in cities_data if c['name'].lower() == city_name.lower()), None)
    if not city:
        raise HTTPException(status_code=404, detail='City not found')
    
    buildings = []
    for i in range(min(limit, city['buildings'] // 1000)):
        buildings.append({
            'id': f'building_{city_name}_{i}',
            'coordinates': [[
                city['lng'] + random.uniform(-0.1, 0.1),
                city['lat'] + random.uniform(-0.1, 0.1),
                0
            ], [
                city['lng'] + random.uniform(-0.1, 0.1),
                city['lat'] + random.uniform(-0.1, 0.1),
                random.uniform(5, 50)
            ]],
            'height': random.uniform(5, 50),
            'building_type': random.choice(['residential', 'commercial', 'industrial']),
            'confidence': random.uniform(0.8, 0.99),
            'area_m2': random.uniform(100, 1000)
        })
    
    return {
        'city': city_name,
        'total_buildings': len(buildings),
        'buildings': buildings,
        'generated_at': datetime.now().isoformat()
    }

@app.post('/api/process')
async def start_processing(data: dict):
    city_name = data.get('city_name', 'Birmingham')
    return {
        'job_id': f'job_{int(datetime.now().timestamp())}_{city_name.lower()}',
        'status': 'processing',
        'message': f'3D processing started for {city_name}'
    }

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
