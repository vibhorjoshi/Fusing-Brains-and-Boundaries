# Building Footprint Extraction API

An advanced API for extracting building footprints from satellite imagery using deep learning techniques.

## Features

- **Mask R-CNN Building Detection**: High-accuracy building detection using a custom Mask R-CNN model
- **Multiple Regularization Techniques**: 
  - RANSAC-based Thresholding (RT)
  - Rectangular Regularization (RR)
  - Feature Enhancement Regularization (FER)
- **Reinforcement Learning Fusion**: Adaptive fusion of regularization results using RL
- **Real-time Processing**: WebSocket support for real-time progress updates
- **Scalable Architecture**: Designed for high-throughput processing
- **Interactive Maps**: Integration with map services for visualization
- **Vectorization**: Convert raster masks to vector polygons

## Architecture

The API is built using a hybrid adaptive architecture that combines:

![Hybrid Adaptive Architecture](./Hybrid-Adaptive-Architecture_Main.png)

## Quick Start

### Prerequisites

- Python 3.9 or later
- Docker and Docker Compose (for containerized deployment)
- MongoDB and Redis (included in Docker Compose)
- PyTorch 2.0+

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/building-footprint-api.git
   cd building-footprint-api
   ```

2. Setup the environment:

   **Windows:**
   ```
   scripts\setup.bat
   ```

   **Linux/Mac:**
   ```
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. Edit the `.env` file with your settings:
   ```
   nano .env
   ```

4. Start the API server:

   **Local development:**
   ```
   uvicorn main:app --reload
   ```

   **Docker:**
   ```
   docker-compose up -d
   ```

5. Open the API documentation:
   ```
   http://localhost:8000/docs
   ```

## API Usage

### Processing a Building Footprint Extraction Request

```python
import requests
import json

url = "http://localhost:8000/api/v1/process"

payload = {
    "bounds": {
        "north": 40.7128,
        "south": 40.7028,
        "east": -74.0060,
        "west": -74.0160
    },
    "zoom_level": 18,
    "enable_rt": True,
    "enable_rr": True, 
    "enable_fer": True,
    "enable_rl_fusion": True
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(json.dumps(response.json(), indent=2))
```

### WebSocket Updates

```javascript
const socket = io('http://localhost:8000', {
  path: '/ws/socket.io'
});

socket.on('connect', () => {
  console.log('Connected to websocket');
  
  // Subscribe to job updates
  socket.emit('join_job', { job_id: 'your-job-id' });
});

socket.on('job_progress', (data) => {
  console.log(`Job ${data.job_id}: ${data.stage} - ${data.progress * 100}%`);
});

socket.on('job_completed', (data) => {
  console.log(`Job ${data.job_id} completed!`);
  console.log(`Buildings found: ${data.result.buildings.length}`);
});
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this API in your research, please cite:

```
@article{building-footprint-api,
  title={Hybrid Adaptive Architecture for Building Footprint Extraction},
  author={Your Name},
  journal={Journal of Geospatial Research},
  year={2023}
}
```