# Streamlit Dashboard Guide

This guide provides instructions for using the GeoAI Streamlit Dashboard for building footprint and crop detection analysis.

## Getting Started

### Launch the Dashboard

1. Make sure all dependencies are installed:
   ```bash
   pip install -r streamlit_requirements.txt
   ```

2. Launch the dashboard using the provided script:
   ```bash
   python launch_streamlit.py
   ```
   
   Alternatively, you can run Streamlit directly:
   ```bash
   streamlit run streamlit_app.py
   ```

3. The dashboard will open in your default web browser at `http://localhost:8501`

## Dashboard Features

The dashboard includes several key sections:

### 1. Live Pipeline

- **Live metrics**: Shows real-time processing statistics
- **Pipeline visualization**: Displays the current stages of the GeoAI pipeline
- **Active jobs monitor**: Lists and tracks ongoing processing jobs

### 2. Alabama Overview

- **Map visualization**: Shows building footprint detection coverage across Alabama
- **City statistics**: Provides detailed metrics for each city

### 3. Analytics

- **Detection accuracy**: Bar chart comparing accuracy across cities
- **Building distribution**: Pie chart showing building distribution

### 4. 3D Visualization

- **3D building footprint**: Interactive 3D visualization of detected building footprints

### 5. Crop Detection (New Feature!)

- **Live crop detection**: Analyze agricultural areas for crop types
- **Region selection**: Choose from different Alabama agricultural regions
- **Analysis options**: Configure the detection parameters
- **Historical trends**: View crop distribution over time
- **Agricultural regions map**: Interactive map of major agricultural areas

## Using the Crop Detection Feature

1. Select "Crop Detection" from the sidebar menu
2. Choose an agricultural region from the dropdown
3. Configure detection settings:
   - Select detection type (Basic, Advanced, or Full Analysis)
   - Toggle crop health and yield estimation options
4. Click "Analyze Crops" to run the detection
5. View results:
   - Crop distribution chart
   - Agricultural land percentage gauge
   - Recent detection history
   - Historical trends

## Troubleshooting

### API Connection Issues

If the API status shows "Backend Disconnected":
- The application will automatically switch to demo mode
- You'll see simulated results rather than live data
- Check that the backend API server is running properly

### GeoAI Library Connection Issues

If you see "GeoAI Library Error":
- Check that the GeoAI library is properly installed
- Verify that you have the necessary Python paths set correctly
- Ensure crop detection module is available

## Development Notes

- The application auto-refreshes every 5 seconds by default
- You can toggle auto-refresh in the sidebar
- New features are being continuously added to improve analysis capabilities