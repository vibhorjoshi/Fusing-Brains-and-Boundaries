# Enhanced Dashboard Integration Guide

This guide explains how to integrate the new components into the existing GeoAI Research Platform HTML dashboard.

## Components to Integrate

1. **Alabama City Performance Analysis** - Google Maps-based visualization
2. **Live Pipeline Visualization** - Real-time pipeline processing visualization 
3. **Backend API Integration** - Connection to the FastAPI backend

## Integration Steps

### Step 1: Add Required JavaScript Files

Add the following script tags to your HTML file just before the closing `</body>` tag:

```html
<!-- Google Maps API (Replace YOUR_API_KEY with your actual Google Maps API key) -->
<script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=visualization"></script>

<!-- Custom JavaScript components -->
<script src="alabama_cities_map.js"></script>
<script src="live_pipeline_visualizer.js"></script>
```

### Step 2: Add Container Elements

Add the following HTML elements where you want the components to appear:

```html
<!-- Alabama City Performance Map -->
<div class="dashboard-card">
    <div class="card-header">
        <span class="card-icon">🗺️</span>
        <h3 class="card-title">Alabama City Performance</h3>
    </div>
    <div id="alabama-map-container" class="map-container" style="height: 500px;"></div>
</div>

<!-- Live Pipeline Visualization -->
<div class="dashboard-card">
    <div class="card-header">
        <span class="card-icon">⚙️</span>
        <h3 class="card-title">Live Processing Pipeline</h3>
    </div>
    <div id="pipeline-visualizer-container"></div>
</div>
```

### Step 3: Initialize Components

Add the following JavaScript to initialize the components:

```html
<script>
    // Initialize the Alabama Map
    document.addEventListener('DOMContentLoaded', function() {
        // Replace with your Google Maps API key
        const apiKey = 'YOUR_GOOGLE_MAPS_API_KEY';
        
        // Initialize the Alabama Map
        initializeAlabamaMap();
        
        // Initialize the Pipeline Visualizer
        const pipelineVisualizer = new LivePipelineVisualizer('#pipeline-visualizer-container', {
            animationDuration: 1000,
            showMetrics: true,
            autoUpdate: true,
            updateInterval: 5000
        });
        
        // Start the pipeline visualization
        pipelineVisualizer.startPipeline();
    });
</script>
```

### Step 4: Add Backend API Integration

Ensure your API connection is properly configured in the JavaScript code:

```javascript
// API configuration
const API_BASE_URL = 'http://localhost:8002';
const API_ENDPOINTS = {
    pipelineStatus: '/api/v1/pipeline/status',
    cityPerformance: '/api/v1/analytics/alabama_performance',
    detectionResults: '/api/v1/building/detect'
};

// Function to make API calls
async function makeAPIRequest(endpoint, method = 'GET', data = null, apiKey = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (apiKey) {
        options.headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    if (data && (method === 'POST' || method === 'PUT')) {
        options.body = JSON.stringify(data);
    }
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    
    if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
    }
    
    return await response.json();
}
```

### Step 5: Register Alabama City Analytics Backend

Add the following line to `unified_backend.py`:

```python
# Import and register the Alabama city analytics router
from alabama_city_analytics import router as alabama_router
app.include_router(alabama_router)
```

### Step 6: Update CSS Styles

Make sure the following CSS is included in your stylesheet:

```css
/* Map Container Styles */
.map-container {
    height: 500px;
    width: 100%;
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid rgba(0, 212, 255, 0.3);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

/* Pipeline Visualizer Styles */
.pipeline-step {
    background: rgba(26, 26, 46, 0.6);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
}

.pipeline-step.active {
    border-color: var(--info-cyan);
    background: rgba(26, 26, 46, 0.8);
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
}

.pipeline-step.complete {
    border-color: var(--success-green);
}

.pipeline-step.error {
    border-color: var(--nasa-red);
}

.pipeline-image-container {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding: 10px 0;
}

.pipeline-image {
    width: 150px;
    height: 150px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    object-fit: cover;
}
```

## Testing Your Integration

1. Start the backend server:
   ```
   python unified_backend.py
   ```

2. Start a simple HTTP server on port 3003:
   ```
   python -m http.server 3003
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3003
   ```

4. Verify that the Alabama city map and pipeline visualization are displaying correctly

## Troubleshooting

### Alabama Map Not Displaying

- Check that your Google Maps API key is valid and has the required permissions
- Open your browser's developer console to check for JavaScript errors
- Verify that the `alabama-map-container` element exists in the DOM

### Pipeline Visualization Not Working

- Ensure the backend server is running and accessible
- Check that the API endpoints are responding with the expected data
- Verify that the `pipeline-visualizer-container` element exists in the DOM

### Backend API Connection Issues

- Ensure the backend server is running on the expected port
- Check CORS settings in the backend server
- Verify API endpoint URLs in the frontend code

## Publishing Changes

After integrating all components, publish the changes to GitHub:

```
python publish_to_github.py
```

This will detect all modified files and push them to the repository.