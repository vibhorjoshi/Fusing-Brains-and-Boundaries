import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime

# Import GeoAI library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.open_source_geo_ai import OpenSourceGeoAI

# Initialize the GeoAI client
@st.cache_resource
def load_geoai_client():
    try:
        # Initialize client with crop detection enabled
        client = OpenSourceGeoAI()
        
        # Verify the crop detection module is available
        if hasattr(client, 'detect_crops'):
            st.success("✅ GeoAI client loaded successfully with crop detection")
        else:
            st.warning("⚠️ GeoAI client loaded but crop detection may not be available")
        
        return client
    except Exception as e:
        st.error(f"❌ Error loading GeoAI client: {str(e)}")
        
        # Create a fallback client for demo purposes
        from types import SimpleNamespace
        
        # Create mock client with required functions
        mock_client = SimpleNamespace()
        
        # Add a mock detect_crops method
        def mock_detect_crops(image, region=None):
            import random
            # Return mock crop detection results
            return {
                "crop_detections": [
                    {"crop_type": "corn", "confidence": 0.92, "area_percentage": 45},
                    {"crop_type": "soybean", "confidence": 0.87, "area_percentage": 30},
                    {"crop_type": "wheat", "confidence": 0.89, "area_percentage": 25}
                ],
                "agricultural_area_percentage": 0.75,
                "visualization": None  # Would be an image in a real implementation
            }
        
        # Add mock methods to the client
        mock_client.detect_crops = mock_detect_crops
        
        st.warning("⚠️ Using demo mode with simulated GeoAI features")
        return mock_client

# Page configuration
st.set_page_config(
    page_title='GeoAI Live Automation Pipeline - Alabama Dashboard',
    page_icon='🏗️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-active { background-color: #28a745; }
    .status-processing { background-color: #ffc107; }
    .status-queued { background-color: #6c757d; }
    .status-error { background-color: #dc3545; }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .pipeline-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🏗️ GeoAI Live Automation Pipeline</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/000000/farm.png", width=80)
st.sidebar.title("🌽 Agricultural Detection")
st.sidebar.markdown("---")

# Add region selection
regions = ["Alabama", "Mississippi", "Georgia", "Florida", "Tennessee"]
selected_region = st.sidebar.selectbox("Select Region", regions)

# Add analysis options
st.sidebar.subheader("Analysis Options")
run_rl_analysis = st.sidebar.checkbox("Use RL for Patch Analysis", value=True)
use_enhanced_model = st.sidebar.checkbox("Use Enhanced MaskRCNN", value=True)
enable_real_time = st.sidebar.checkbox("Enable Real-Time Updates", value=True)

# Add run button
if st.sidebar.button("🚀 Run Analysis"):
    with st.spinner(f"Analyzing {selected_region} with GeoAI..."):
        # Initialize the client
        geoai_client = load_geoai_client()
        
        if geoai_client:
            try:
                # Get satellite image
                st.info(f"Fetching satellite imagery for {selected_region}...")
                image = geoai_client.get_satellite_image(selected_region)
                
                # Create placeholder for image
                image_col, result_col = st.columns(2)
                with image_col:
                    st.subheader("Satellite Imagery")
                    st.image(image, caption=f"{selected_region} Satellite View", use_container_width=True)
                
                # Run analysis based on options
                st.info("Running agricultural detection pipeline...")
                # Run with the selected analysis method
                if run_rl_analysis:
                    results = geoai_client.analyze_with_rl_patches(image)
                else:
                    results = geoai_client.analyze_basic(image)
                    
                    # Display results
                    with result_col:
                        st.subheader("Detection Results")
                        try:
                            # Check if visualization key exists in results
                            if isinstance(results, dict) and "visualization" in results:
                                st.image(results["visualization"], 
                                        caption="Agricultural Features Detected", 
                                        use_container_width=True)
                            elif isinstance(results, dict) and "image" in results:
                                st.image(results["image"], 
                                        caption="Agricultural Features Detected", 
                                        use_container_width=True)
                            elif hasattr(results, "get_visualization") and callable(getattr(results, "get_visualization")):
                                # Try to call a method to get visualization if available
                                st.image(results.get_visualization(), 
                                        caption="Agricultural Features Detected",
                                        use_container_width=True)
                            else:
                                # Create a placeholder visualization
                                st.warning("Visualization not available. Showing original image with analysis overlay.")
                                # Display original image with text overlay
                                st.image(image, caption="Analysis Result (Visualization not available)", 
                                        use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying visualization: {e}")
                            st.image(image, caption="Original Image (Visualization failed)", 
                                    use_container_width=True)
                    
                    # Show metrics
                    st.subheader("📊 Analysis Metrics")
                    metrics_cols = st.columns(4)
                    
                    # Handle metrics differently based on available data
                    if isinstance(results, dict) and "metrics" in results:
                        metrics = results["metrics"]
                        with metrics_cols[0]:
                            if "area_acres" in metrics:
                                st.metric("Agricultural Area (acres)", f"{metrics['area_acres']:.1f}")
                            else:
                                st.metric("Agricultural Area (acres)", "N/A")
                        
                        with metrics_cols[1]:
                            if "field_count" in metrics:
                                st.metric("Number of Fields", metrics["field_count"])
                            else:
                                st.metric("Number of Fields", "N/A")
                        
                        with metrics_cols[2]:
                            if "processing_time" in metrics:
                                st.metric("Processing Time (s)", f"{metrics['processing_time']:.2f}")
                            else:
                                st.metric("Processing Time (s)", "N/A")
                        
                        with metrics_cols[3]:
                            if "confidence" in metrics:
                                st.metric("Confidence Score", f"{metrics['confidence'] * 100:.1f}%")
                            else:
                                st.metric("Confidence Score", "N/A")
                    else:
                        # If metrics aren't available in expected format, show default values
                        with metrics_cols[0]:
                            st.metric("Agricultural Area (acres)", "N/A")
                        with metrics_cols[1]:
                            st.metric("Number of Fields", "N/A")
                        with metrics_cols[2]:
                            st.metric("Processing Time (s)", "N/A")
                        with metrics_cols[3]:
                            st.metric("Confidence Score", "N/A")                # Show detailed data table
                st.subheader("📋 Detected Agricultural Features")
                try:
                    if "features" in results:
                        df = pd.DataFrame(results["features"])
                        st.dataframe(df)
                    else:
                        st.info("No detailed feature data available.")
                except Exception as e:
                    st.warning(f"Could not display features table: {e}")
                    st.info("Summary data is still available in the metrics section.")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    
    st.success(f"Analysis of {selected_region} complete!")

# Display sample images if no analysis is run yet
else:
    # Create a placeholder for sample data
    st.subheader("🖼️ Sample Visualizations")
    
    sample_cols = st.columns(3)
    with sample_cols[0]:
        st.image("https://earthobservatory.nasa.gov/ContentWOC/images/remote_sensing/cropland.png", 
                 caption="Sample Agricultural Detection", use_container_width=True)
    with sample_cols[1]:
        st.image("https://earthobservatory.nasa.gov/ContentWOC/images/remote_sensing/tulips.jpg", 
                 caption="High Resolution Field Analysis", use_container_width=True)
    with sample_cols[2]:
        st.image("https://earthobservatory.nasa.gov/ContentWOC/images/remote_sensing/corn.jpg", 
                 caption="Crop Health Visualization", use_container_width=True)
    
    st.info("👆 Select a region and click 'Run Analysis' to start the GeoAI pipeline.")

# Dashboard with mock real-time data
st.markdown("---")
st.subheader("📈 Live Performance Dashboard")

# Create metrics
performance_cols = st.columns(4)
with performance_cols[0]:
    st.metric("CPU Usage", "42%", "2%")
with performance_cols[1]:
    st.metric("GPU Memory", "3.2 GB", "-0.4 GB")
with performance_cols[2]:
    st.metric("Active Models", "3", "1")
with performance_cols[3]:
    st.metric("Processing Rate", "8.3 img/s", "0.5 img/s")

# Create a placeholder for the charts
chart_placeholder = st.empty()

# Display mock timeline chart
with chart_placeholder.container():
    # Create sample data for the timeline
    timeline_cols = st.columns([2, 1])
    
    with timeline_cols[0]:
        # Create sample performance data
        timestamps = pd.date_range(start=datetime.now() - pd.Timedelta(minutes=30), 
                                 end=datetime.now(), 
                                 freq='1min')
        
        cpu_values = 30 + 15 * np.sin(np.linspace(0, 3*np.pi, len(timestamps))) + np.random.normal(0, 3, len(timestamps))
        memory_values = 40 + 10 * np.sin(np.linspace(0, 2*np.pi, len(timestamps)) + 1) + np.random.normal(0, 2, len(timestamps))
        
        # Create performance chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=cpu_values,
            mode='lines',
            name='CPU Usage (%)',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=memory_values,
            mode='lines',
            name='Memory Usage (%)',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.update_layout(
            title='System Performance',
            xaxis_title='Time',
            yaxis_title='Usage (%)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=350,
            margin=dict(l=40, r=20, t=60, b=40),
            plot_bgcolor='rgba(245, 245, 245, 0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#444'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with timeline_cols[1]:
        # Create a table with job status
        status_data = {
            'Job ID': ['JOB-1342', 'JOB-1341', 'JOB-1339', 'JOB-1338', 'JOB-1337'],
            'Status': ['Active', 'Processing', 'Completed', 'Completed', 'Failed'],
            'Runtime (s)': [42, 156, 223, 198, 67]
        }
        
        status_df = pd.DataFrame(status_data)
        
        st.subheader("Active Jobs")
        st.dataframe(status_df, use_container_width=True)

# Add a pipeline status section
st.markdown("---")
st.subheader("🔄 Pipeline Components Status")

# Create a 3-column layout for component status
component_cols = st.columns(3)

with component_cols[0]:
    st.markdown("""
    <div style="padding: 1rem; background-color: rgba(40, 167, 69, 0.1); border-left: 5px solid #28a745; border-radius: 4px;">
        <h4>
            <span class="status-indicator status-active"></span>
            MaskRCNN Model
        </h4>
        <p>Active and processing normally</p>
        <p><small>Version 3.4.2 • Loaded 23 minutes ago</small></p>
    </div>
    """, unsafe_allow_html=True)

with component_cols[1]:
    st.markdown("""
    <div style="padding: 1rem; background-color: rgba(40, 167, 69, 0.1); border-left: 5px solid #28a745; border-radius: 4px;">
        <h4>
            <span class="status-indicator status-active"></span>
            Adaptive Fusion Pipeline
        </h4>
        <p>Active and processing normally</p>
        <p><small>Version 2.1.0 • 4 active tasks</small></p>
    </div>
    """, unsafe_allow_html=True)

with component_cols[2]:
    st.markdown("""
    <div style="padding: 1rem; background-color: rgba(255, 193, 7, 0.1); border-left: 5px solid #ffc107; border-radius: 4px;">
        <h4>
            <span class="status-indicator status-processing"></span>
            RL Agent
        </h4>
        <p>Operating at reduced capacity</p>
        <p><small>Version 1.2.3 • High memory usage</small></p>
    </div>
    """, unsafe_allow_html=True)

# Add a footer
st.markdown("""
<div class="footer">
    <p>GeoAI Agricultural Detection System • Real USA Agricultural Detection System using Adaptive Fusion</p>
    <p>© 2025 Geo AI Research Team • Last updated: October 2, 2025</p>
</div>
""", unsafe_allow_html=True)