# GeoAI Research Platform - Enhanced HTML Dashboard

This is the documentation for the enhanced HTML frontend dashboard for the GeoAI Building Footprint Detection research platform. The dashboard provides an interactive visualization interface for exploring building footprint detection results, performance metrics, and geospatial analysis.

## Features

### NASA-Themed Interface
The dashboard follows a NASA-inspired design language with a dark space-themed UI and professional data visualization components. The interface includes:

- Real-time status indicators for all system components
- Advanced performance metrics visualization
- Interactive geospatial mapping for Alabama cities
- Live pipeline visualization showing each step of the processing
- API testing interface for exploring backend functionality

### Live Pipeline Visualization
The dashboard includes a real-time visualization of the building footprint detection pipeline, showing:

1. Initial building detection
2. RT (Regular Topology) regularization
3. RR (Regular Rectangle) regularization 
4. FER (Feature Edge Regularization)
5. RL Adaptive Fusion process
6. LapNet Refinement stage
7. Final output visualization

### Alabama City Performance Analysis
An interactive Google Maps-based visualization shows building footprint detection performance across major Alabama cities:

- IoU (Intersection over Union) scores
- Building detection counts
- Processing time metrics
- Precision and recall measurements
- Heatmap visualization of performance metrics

### API Integration
The dashboard connects to the FastAPI backend running on port 8002, providing:

- Secure authenticated API access
- Real-time data polling
- Performance metrics visualization
- Interactive API testing interface

## Technical Implementation

The dashboard is built using:

- HTML5 for structure
- CSS3 with custom animations and NASA-inspired styling
- Vanilla JavaScript for interactivity
- Custom visualization components:
  - `alabama_cities_map.js` - Google Maps integration for Alabama cities
  - `live_pipeline_visualizer.js` - Real-time pipeline visualization

No external frameworks are required, making the dashboard lightweight and fast to load.

## Getting Started

1. Clone the repository:
```
git clone https://github.com/your-username/geo-ai-research-paper.git
```

2. Navigate to the project directory:
```
cd geo-ai-research-paper
```

3. Start the backend API server:
```
python unified_backend.py
```

4. Start a simple HTTP server to serve the dashboard:
```
python -m http.server 3003
```

5. Open your browser and navigate to:
```
http://localhost:3003
```

## Publishing to GitHub

To publish any unpublished files to GitHub, use the included `publish_to_github.py` script:

```
python publish_to_github.py
```

This will identify any files that haven't been committed and push them to the GitHub repository.

## Customizing the Dashboard

### Adding New Visualizations

1. Create a new JavaScript file for your visualization component
2. Import the file in `index.html`
3. Initialize your visualization in the appropriate container

### Modifying Styles

The dashboard uses CSS variables for consistent styling. Major theme colors include:

```css
:root {
    --nasa-blue: #0B3D91;
    --nasa-red: #FC3D21;
    --space-black: #0a0a0a;
    --deep-space: #1a1a2e;
    --cosmic-blue: #16213e;
    --success-green: #00ff88;
    --warning-orange: #ff6b35;
    --info-cyan: #00d4ff;
    --star-white: #eee6ff;
}
```

## Contributing

Contributions to improve the dashboard are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.