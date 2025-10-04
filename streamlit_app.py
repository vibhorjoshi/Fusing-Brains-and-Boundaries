# Fix matplotlib and fontconfig permissions for containerized environments
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['FONTCONFIG_PATH'] = '/tmp/fontconfig'
os.environ['FONTCONFIG_FILE'] = '/tmp/fontconfig/fonts.conf'

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import queue
from datetime import datetime, timedelta
import random
import json
import sys
from PIL import Image
import io

# Try importing plotly, if fails show error
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Plotly not installed. Please add 'plotly>=5.17.0' to requirements.txt")
    st.stop()

# Try importing requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    st.warning("⚠️ Requests library not available. API features disabled.")

# Page configuration
st.set_page_config(
    page_title='GeoAI Live Automation Pipeline - Alabama Dashboard',
    page_icon='🏗️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
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
    .metric-live {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Global state for live processing
if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = queue.Queue()
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}
if 'live_metrics' not in st.session_state:
    st.session_state.live_metrics = {
        'total_processed': 0,
        'success_rate': 94.2,
        'avg_processing_time': 2.3,
        'active_jobs': 0
    }

def get_api_data(endpoint):
    """Fetch data from FastAPI backend"""
    if not REQUESTS_AVAILABLE:
        return get_mock_data(endpoint)
    
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return get_mock_data(endpoint)
    except:
        return get_mock_data(endpoint)

def get_mock_data(endpoint):
    """Return mock data when API is unavailable"""
    if endpoint == "/":
        return {"status": "simulated", "message": "Running in demo mode"}
    elif endpoint == "/api/cities":
        return [
            {"name": "Birmingham", "lat": 33.5186, "lng": -86.8104, "buildings": 45230, "accuracy": 94.2},
            {"name": "Montgomery", "lat": 32.3792, "lng": -86.3077, "buildings": 32150, "accuracy": 92.8},
            {"name": "Mobile", "lat": 30.6954, "lng": -88.0399, "buildings": 28940, "accuracy": 93.5},
            {"name": "Huntsville", "lat": 34.7304, "lng": -86.5861, "buildings": 38670, "accuracy": 95.1},
            {"name": "Tuscaloosa", "lat": 33.2098, "lng": -87.5692, "buildings": 22480, "accuracy": 91.7}
        ]
    return None

def create_live_metrics_dashboard():
    """Create live metrics dashboard"""
    st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🏗️ Total Processed",
            f"{st.session_state.live_metrics['total_processed']:,}",
            "+1"
        )

    with col2:
        st.metric(
            "📊 Success Rate",
            f"{st.session_state.live_metrics['success_rate']:.1f}%",
            "+0.1%"
        )

    with col3:
        st.metric(
            "⚡ Avg Processing Time",
            f"{st.session_state.live_metrics['avg_processing_time']:.1f}s",
            "-0.2s"
        )

    with col4:
        st.metric(
            "🔄 Active Jobs",
            st.session_state.live_metrics['active_jobs'],
            "↗️ 1"
        )
    st.markdown('</div>', unsafe_allow_html=True)

def create_pipeline_visualization():
    """Create live pipeline visualization"""
    st.subheader("🔄 Live Processing Pipeline")

    stages = ["Input", "Preprocessing", "AI Detection", "Post-processing", "Output"]
    stage_status = ["completed", "completed", "processing", "queued", "queued"]

    cols = st.columns(len(stages))
    for i, (stage, status) in enumerate(zip(stages, stage_status)):
        with cols[i]:
            if status == "completed":
                st.success(f"✅ {stage}")
            elif status == "processing":
                st.info(f"🔄 {stage}")
            else:
                st.warning(f"⏳ {stage}")

def create_active_jobs_monitor():
    """Create active jobs monitoring table"""
    st.subheader("📋 Active Processing Jobs")

    active_jobs = [
        job for job in st.session_state.processing_status.values()
        if job['status'] in ['queued', 'processing']
    ]

    if active_jobs:
        for job in active_jobs[-5:]:
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 1])

            with col1:
                st.write(f"**{job['city']}**")

            with col2:
                status_class = {
                    'processing': 'status-processing',
                    'queued': 'status-queued',
                    'completed': 'status-active',
                    'error': 'status-error'
                }.get(job['status'], 'status-queued')

                st.markdown(f"""
                <div style="display: flex; align-items: center;">
                    <div class="status-indicator {status_class}"></div>
                    {job['status'].title()}
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.progress(job['progress'] / 100)

            with col4:
                st.write(f"{job['progress']}%")

            with col5:
                st.write(f"{job['eta']}s")
    else:
        st.info("No active jobs currently processing")

def create_alabama_overview():
    """Create Alabama cities overview"""
    st.subheader("🗺️ Alabama Cities Overview")

    cities_data = get_api_data("/api/cities")
    if cities_data:
        df = pd.DataFrame(cities_data)

        # Map visualization
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lng",
            size="buildings",
            color="accuracy",
            hover_name="name",
            hover_data=["buildings", "accuracy"],
            color_continuous_scale="Viridis",
            size_max=50,
            zoom=6,
            center={"lat": 32.8067, "lon": -86.7911},
            title="Alabama Building Footprint Detection Coverage"
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":50,"l":0,"b":0},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cities table
        st.subheader("📊 City Statistics")
        st.dataframe(
            df[['name', 'buildings', 'accuracy']].rename(columns={
                'name': 'City',
                'buildings': 'Buildings Detected',
                'accuracy': 'Accuracy (%)'
            }),
            use_container_width=True
        )

def create_analytics_dashboard():
    """Create analytics dashboard"""
    st.subheader("📈 Performance Analytics")

    cities_data = get_api_data("/api/cities")
    if cities_data:
        df = pd.DataFrame(cities_data)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                df,
                x='name',
                y='accuracy',
                title='Detection Accuracy by City',
                labels={'name': 'City', 'accuracy': 'Accuracy (%)'},
                color='accuracy',
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                df,
                values='buildings',
                names='name',
                title='Building Distribution',
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def create_3d_visualization():
    """Create 3D building visualization"""
    st.subheader("🏗️ 3D Building Footprint Visualization")

    buildings_3d = []
    for i in range(50):
        buildings_3d.append({
            'x': np.random.uniform(-200, 200),
            'y': np.random.uniform(-200, 200),
            'z': np.random.uniform(0, 100),
            'height': np.random.uniform(5, 100),
            'type': np.random.choice(['residential', 'commercial', 'industrial']),
            'confidence': np.random.uniform(0.7, 0.99)
        })

    df_3d = pd.DataFrame(buildings_3d)

    fig = go.Figure(data=[go.Scatter3d(
        x=df_3d['x'],
        y=df_3d['y'],
        z=df_3d['z'],
        mode='markers',
        marker=dict(
            size=df_3d['height']/3,
            color=df_3d['confidence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        text=[f"Height: {h:.1f}m<br>Type: {t}<br>Confidence: {c:.2f}"
              for h, t, c in zip(df_3d['height'], df_3d['type'], df_3d['confidence'])],
        hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>%{text}"
    )])

    fig.update_layout(
        title="3D Building Footprint Detection Results",
        scene=dict(
            xaxis_title='Longitude Offset (m)',
            yaxis_title='Latitude Offset (m)',
            zaxis_title='Height (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

def create_crop_detection_page():
    """Create crop detection dashboard"""
    st.subheader("🌾 Crop Detection Dashboard")
    
    st.info("🚧 Crop detection feature coming soon! This module will integrate with the GeoAI library for agricultural analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Select Region")
        regions = ["Alabama - Central Valley", "Alabama - Black Belt", 
                  "Alabama - Tennessee Valley", "Alabama - Gulf Coast"]
        selected_region = st.selectbox("Region", regions)
        
        st.markdown("### ⚙️ Detection Settings")
        detection_type = st.radio("Type", ["Basic", "Advanced", "Full Analysis"])
        
    with col2:
        st.markdown("### 📊 Recent Results")
        recent_data = pd.DataFrame({
            "Region": ["Central Valley", "Gulf Coast", "Black Belt"],
            "Date": ["2024-10-01", "2024-09-28", "2024-09-25"],
            "Primary Crop": ["Corn (65%)", "Cotton (48%)", "Soybeans (54%)"]
        })
        st.dataframe(recent_data, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 GeoAI Live Automation Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Building Footprint and Crop Detection for Alabama Cities")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")

        # API Status
        api_status = get_api_data("/")
        if api_status and api_status.get('status') != 'simulated':
            st.success("✅ Backend Connected")
        else:
            st.warning("⚠️ Demo Mode")
            st.caption("Running with simulated data")

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Select Dashboard:",
            ["Live Pipeline", "Alabama Overview", "Analytics", "3D Visualization", "Crop Detection"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.caption("💡 Tip: Refresh to see updated metrics")

    # Main content
    if page == "Live Pipeline":
        create_live_metrics_dashboard()
        create_pipeline_visualization()
        create_active_jobs_monitor()

    elif page == "Alabama Overview":
        create_alabama_overview()

    elif page == "Analytics":
        create_analytics_dashboard()

    elif page == "3D Visualization":
        create_3d_visualization()
        
    elif page == "Crop Detection":
        create_crop_detection_page()

    # Footer
    st.markdown("---")
    st.markdown("### 🛰️ Powered by NASA GeoAI Technology | Real-time Building Footprint & Crop Detection")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
