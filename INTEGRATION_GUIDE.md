# GeoAI Integration and Extension Guide

This guide provides instructions for integrating with and extending the GeoAI Research Project with additional data sources, new visualizations, and custom processing pipelines.

## Table of Contents

- [API Integration](#api-integration)
- [Adding New Data Sources](#adding-new-data-sources)
- [Custom Visualization Development](#custom-visualization-development)
- [Extending the GeoAI Engine](#extending-the-geoai-engine)
- [Contributing Guidelines](#contributing-guidelines)

## API Integration

### REST API Endpoints

The GeoAI system provides a comprehensive REST API for integration with other applications.

#### Base URL

- Development: `http://localhost:8000/api/v1`
- Production: `https://your-api-domain.com/api/v1`

#### Authentication

```
Authorization: Bearer <your_api_token>
```

#### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/regions` | GET | List all available regions |
| `/regions/{id}/analyze` | POST | Analyze a specific region |
| `/buildings` | GET | Query detected buildings |
| `/visualize/{id}` | GET | Generate visualization |

### Python Client Example

```python
import requests

API_URL = "http://localhost:8000/api/v1"
API_TOKEN = "your_api_token"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# List regions
response = requests.get(f"{API_URL}/regions", headers=headers)
regions = response.json()

# Analyze a region
region_id = regions[0]["id"]
analysis_params = {
    "detail_level": "high",
    "include_metadata": True
}

response = requests.post(
    f"{API_URL}/regions/{region_id}/analyze", 
    headers=headers,
    json=analysis_params
)

results = response.json()
```

## Adding New Data Sources

The GeoAI system can be extended to include additional data sources beyond the default satellite imagery.

### Supported Data Source Types

- Satellite imagery (GeoTIFF, JP2)
- Vector data (GeoJSON, Shapefile)
- Tabular data with spatial components (CSV with lat/long)
- LiDAR point clouds

### Steps to Add a New Data Source

1. Place your data files in the `data/sources/{source_name}` directory
2. Create a data source configuration file:

```json
// data/sources/my_new_source/config.json
{
  "name": "My New Data Source",
  "type": "satellite_imagery",
  "format": "geotiff",
  "coordinate_system": "EPSG:4326",
  "temporal_range": {
    "start": "2023-01-01",
    "end": "2023-12-31"
  },
  "preprocessing": {
    "normalize": true,
    "resample_resolution": 0.5
  }
}
```

3. Implement a source adapter in `src/data/adapters/my_source_adapter.py`:

```python
from src.data.base_adapter import BaseAdapter

class MySourceAdapter(BaseAdapter):
    def __init__(self, config_path):
        super().__init__(config_path)
        
    def load_data(self, region_id, **kwargs):
        # Implementation for loading your specific data format
        pass
        
    def preprocess(self, data):
        # Implementation for preprocessing your data
        pass
```

4. Register your adapter in `src/data/adapter_registry.py`

## Custom Visualization Development

### Frontend Visualizations

To add a new visualization component to the frontend:

1. Create a new component in `frontend/components/visualizations/`
2. Use our visualization hooks:

```jsx
// frontend/components/visualizations/MyCustomVisualization.jsx
import React from 'react';
import { useGeoData, useMapSettings } from '@/hooks/geo';

export const MyCustomVisualization = ({ regionId }) => {
  const { data, isLoading, error } = useGeoData(regionId);
  const { mapSettings, updateMapSettings } = useMapSettings();
  
  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  
  // Your visualization rendering logic
  return (
    <div className="custom-visualization">
      {/* Your visualization component */}
    </div>
  );
};
```

3. Register your visualization in `frontend/config/visualizations.js`

### Backend Visualization Pipelines

To add a new backend visualization generator:

1. Create a new module in `src/visualization/generators/`
2. Implement the BaseVisGenerator interface:

```python
# src/visualization/generators/my_custom_vis.py
from src.visualization.base_generator import BaseVisGenerator
import matplotlib.pyplot as plt

class MyCustomVisGenerator(BaseVisGenerator):
    def __init__(self, config=None):
        super().__init__(config)
        
    def generate(self, geo_data, output_path=None, **kwargs):
        # Your visualization generation code
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Process and visualize geo_data
        # ...
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return fig
```

3. Register your generator in `src/visualization/generator_registry.py`

## Extending the GeoAI Engine

### Custom Model Integration

To integrate a custom deep learning model:

1. Place your model code in `src/models/custom/`
2. Create a wrapper that implements the BaseModel interface:

```python
# src/models/custom/my_custom_model.py
from src.models.base_model import BaseModel
import torch

class MyCustomModel(BaseModel):
    def __init__(self, model_path, config=None):
        super().__init__(config)
        self.model = torch.load(model_path)
        
    def preprocess(self, input_data):
        # Preprocessing logic
        return processed_data
        
    def predict(self, processed_data):
        # Inference logic
        with torch.no_grad():
            outputs = self.model(processed_data)
        return outputs
        
    def postprocess(self, outputs):
        # Convert raw outputs to structured results
        return results
```

3. Register your model in `src/models/model_registry.py`

### Custom Processing Pipeline

To create a new processing pipeline:

1. Define your pipeline in `src/pipelines/`:

```python
# src/pipelines/my_custom_pipeline.py
from src.pipelines.base_pipeline import BasePipeline

class MyCustomPipeline(BasePipeline):
    def __init__(self, config=None):
        super().__init__(config)
        
    def setup(self):
        # Initialize components
        self.data_loader = self.get_component('data_loader')
        self.model = self.get_component('model')
        self.visualizer = self.get_component('visualizer')
        
    def run(self, input_data):
        # Pipeline execution logic
        loaded_data = self.data_loader.load(input_data)
        processed_data = self.data_loader.preprocess(loaded_data)
        model_output = self.model.predict(processed_data)
        results = self.model.postprocess(model_output)
        
        if self.visualizer:
            visualization = self.visualizer.generate(results)
            results['visualization'] = visualization
            
        return results
```

2. Register your pipeline in `unified_config.py`

## Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -am 'Add new feature'`)
7. Push to the branch (`git push origin feature/my-feature`)
8. Create a new Pull Request

Please ensure your code follows our coding standards and includes appropriate documentation.

---

For additional support or questions, please refer to the documentation or open an issue on GitHub.