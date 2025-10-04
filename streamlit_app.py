# Fix matplotlib and fontconfig permissions for containerized environments
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['FONTCONFIG_PATH'] = '/tmp/fontconfig'
os.environ['FONTCONFIG_FILE'] = '/tmp/fontconfig/fonts.conf'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import threading
import queue
from datetime import datetime, timedelta
import random
import json
import sys
import os
from PIL import Image
import io

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
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def simulate_live_processing():
    """Simulate live processing pipeline"""
    cities = ['Birmingham', 'Montgomery', 'Mobile', 'Huntsville', 'Tuscaloosa']

    while True:
        # Simulate new processing job
        city = random.choice(cities)
        job_id = f"job_{int(time.time())}_{city.lower()}"

        st.session_state.processing_status[job_id] = {
            'city': city,
            'status': 'queued',
            'progress': 0,
            'start_time': datetime.now(),
            'eta': random.randint(30, 120)
        }

        # Process the job
        for progress in range(0, 101, 10):
            st.session_state.processing_status[job_id]['progress'] = progress
            if progress < 30:
                st.session_state.processing_status[job_id]['status'] = 'queued'
            elif progress < 80:
                st.session_state.processing_status[job_id]['status'] = 'processing'
            else:
                st.session_state.processing_status[job_id]['status'] = 'completed'

            time.sleep(2)  # Simulate processing time

        # Update live metrics
        st.session_state.live_metrics['total_processed'] += 1
        st.session_state.live_metrics['active_jobs'] = len([
            job for job in st.session_state.processing_status.values()
            if job['status'] in ['queued', 'processing']
        ])

        time.sleep(random.randint(10, 30))  # Wait before next job

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

    # Pipeline stages
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
        for job in active_jobs[-5:]:  # Show last 5 jobs
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
            # Accuracy comparison
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
            # Buildings distribution
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

    # Generate mock 3D building data
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

    # 3D scatter plot
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
        if api_status:
            st.success("✅ Backend Connected")
            st.caption(f"Status: {api_status.get('status', 'Unknown')}")
        else:
            st.error("❌ Backend Disconnected")
            st.caption("Running in fallback mode with simulated data")

        # Load GeoAI client status from streamlit_backend
        try:
            # Import GeoAI library
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.open_source_geo_ai import OpenSourceGeoAI
            from streamlit_backend import load_geoai_client
            
            # Check GeoAI client
            geoai_client = load_geoai_client()
            if geoai_client:
                if hasattr(geoai_client, 'detect_crops'):
                    st.success("✅ GeoAI Library Connected")
                    st.caption("Crop detection available")
                else:
                    st.warning("⚠️ GeoAI Library Partial")
                    st.caption("Crop detection unavailable")
            else:
                st.error("❌ GeoAI Library Disconnected")
                st.caption("Running in simulated mode")
        except Exception as e:
            st.error("❌ GeoAI Library Error")
            st.caption(f"Error: {str(e)[:50]}...")

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Select Dashboard:",
            ["Live Pipeline", "Alabama Overview", "Analytics", "3D Visualization", "Crop Detection"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Pipeline Controls
        st.subheader("⚙️ Pipeline Controls")

        if st.button("🔄 Start Live Processing", type="primary"):
            if 'processing_thread' not in st.session_state:
                st.session_state.processing_thread = threading.Thread(
                    target=simulate_live_processing,
                    daemon=True
                )
                st.session_state.processing_thread.start()
                st.success("Live processing started!")
            else:
                st.warning("Live processing already running!")

        if st.button("⏹️ Stop Processing"):
            # Note: In a real app, you'd have proper thread management
            st.info("Processing will continue until page refresh")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)

    # Main content based on selected page
    if page == "Live Pipeline":
        # Live metrics
        create_live_metrics_dashboard()

        # Pipeline visualization
        create_pipeline_visualization()

        # Active jobs monitor
        create_active_jobs_monitor()

    elif page == "Alabama Overview":
        create_alabama_overview()

    elif page == "Analytics":
        create_analytics_dashboard()

    elif page == "3D Visualization":
        create_3d_visualization()
        
    elif page == "Crop Detection":
        st.subheader("🌾 Live Crop Detection Dashboard")
        
        # Import crop detection functionality
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from streamlit_backend import load_geoai_client
            
            # Get GeoAI client
            geoai_client = load_geoai_client()
            
            # Set up layout for crop detection
            col1, col2 = st.columns(2)
            
            with col1:
                # Region selection
                regions = ["Alabama - Central Valley", "Alabama - Black Belt", 
                          "Alabama - Tennessee Valley", "Alabama - Gulf Coast", 
                          "Alabama - Piedmont Region"]
                selected_region = st.selectbox("Select Agricultural Region", regions)
                
                # Analysis options
                st.subheader("Analysis Settings")
                detection_type = st.radio("Detection Type", 
                                        ["Basic Detection", "Advanced Classification", "Full Analysis"])
                
                include_health = st.checkbox("Include Crop Health Assessment", value=True)
                include_yield = st.checkbox("Include Yield Estimation", value=True)
                
                # Run analysis button
                if st.button("🔍 Analyze Crops", type="primary"):
                    if geoai_client:
                        with st.spinner(f"Analyzing crops in {selected_region}..."):
                            try:
                                # Get mock image for the selected region
                                mock_images = {
                                    "Alabama - Central Valley": "https://extension.missouri.edu/media/wysiwyg/Extensiondata/NewsAdmin/Photos/2022/20220512-crops-lg.jpg",
                                    "Alabama - Black Belt": "https://www.agriculture.com/sites/agriculture.com/files/styles/ratio_16_9_ms/public/field/image/353671086_2422-1.jpg",
                                    "Alabama - Tennessee Valley": "https://wallpaperaccess.com/full/2774267.jpg",
                                    "Alabama - Gulf Coast": "https://media.istockphoto.com/id/589274410/photo/corn-field-aerial.jpg?s=612x612&w=0&k=20&c=z9zHLexO_x7kKcQysM4J9bYyV_zEVCZ5W6QrsL9Gxyw=",
                                    "Alabama - Piedmont Region": "https://miro.medium.com/v2/resize:fit:1400/1*RM5WEbMkKd3jECEXFjYsrQ.jpeg"
                                }
                                
                                # Display the image
                                image_url = mock_images.get(selected_region, mock_images["Alabama - Central Valley"])
                                st.image(image_url, caption=f"Satellite imagery of {selected_region}", use_container_width=True)
                                
                                # Run detection with appropriate options
                                if detection_type == "Basic Detection":
                                    complexity = "basic"
                                elif detection_type == "Advanced Classification":
                                    complexity = "advanced"
                                else:
                                    complexity = "full"
                                    
                                # Call the detect_crops method
                                results = geoai_client.detect_crops(
                                    image_url, 
                                    region=selected_region, 
                                )
                                
                                # Display results
                                st.subheader("🌽 Crop Detection Results")
                                
                                # Display crop types and confidence
                                if results and "crop_detections" in results:
                                    crops_df = pd.DataFrame(results["crop_detections"])
                                    st.dataframe(crops_df, use_container_width=True)
                                    
                                    # Create visualization
                                    fig = px.bar(
                                        crops_df,
                                        x="crop_type",
                                        y="area_percentage",
                                        color="confidence",
                                        color_continuous_scale="Viridis",
                                        title="Detected Crops by Area (%)",
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display agricultural percentage
                                    if "agricultural_area_percentage" in results:
                                        agricultural_pct = results["agricultural_area_percentage"] * 100
                                        
                                        # Create gauge chart
                                        fig = go.Figure(go.Indicator(
                                            mode="gauge+number",
                                            value=agricultural_pct,
                                            title={'text': "Agricultural Land (%)"},
                                            domain={'x': [0, 1], 'y': [0, 1]},
                                            gauge={
                                                'axis': {'range': [0, 100]},
                                                'bar': {'color': "green"},
                                                'steps': [
                                                    {'range': [0, 30], 'color': "#ffdd99"},
                                                    {'range': [30, 70], 'color': "#99cc99"},
                                                    {'range': [70, 100], 'color': "#339933"}
                                                ]
                                            }
                                        ))
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("Failed to detect crops in the selected region")
                                
                            except Exception as e:
                                st.error(f"Error during crop detection: {str(e)}")
                    else:
                        st.error("GeoAI client not available. Please check the connection and try again.")
            
            with col2:
                st.subheader("Recent Detection Results")
                
                # Show mock recent detections
                recent_data = [
                    {"region": "Alabama - Central Valley", "date": "2023-10-15", "crops": "Corn (65%), Soybeans (25%)"},
                    {"region": "Alabama - Gulf Coast", "date": "2023-10-14", "crops": "Cotton (48%), Peanuts (32%)"},
                    {"region": "Alabama - Black Belt", "date": "2023-10-12", "crops": "Soybeans (54%), Wheat (31%)"},
                    {"region": "Alabama - Tennessee Valley", "date": "2023-10-10", "crops": "Cotton (58%), Corn (22%)"},
                ]
                
                # Display in a nice table
                st.table(pd.DataFrame(recent_data))
                
                # Historical trend visualization
                st.subheader("Historical Crop Distribution")
                
                # Create mock data for historical trends
                years = [2019, 2020, 2021, 2022, 2023]
                corn = [32, 35, 30, 28, 33]
                soybeans = [28, 30, 35, 38, 40]
                cotton = [25, 22, 20, 18, 15]
                wheat = [15, 13, 15, 16, 12]
                
                # Create a DataFrame
                hist_df = pd.DataFrame({
                    'Year': years,
                    'Corn': corn,
                    'Soybeans': soybeans,
                    'Cotton': cotton,
                    'Wheat': wheat
                })
                
                # Plot the data
                fig = px.line(
                    hist_df, 
                    x='Year', 
                    y=['Corn', 'Soybeans', 'Cotton', 'Wheat'],
                    title="Crop Distribution Over Time (%)",
                    markers=True,
                    line_shape="spline"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a map visualization
                st.subheader("Agricultural Regions")
                
                # Create sample data for regions
                region_data = [
                    {"name": "Central Valley", "lat": 32.9, "lon": -86.6, "area": 1200, "main_crop": "Corn"},
                    {"name": "Black Belt", "lat": 32.3, "lon": -87.4, "area": 950, "main_crop": "Soybeans"},
                    {"name": "Tennessee Valley", "lat": 34.7, "lon": -86.8, "area": 850, "main_crop": "Cotton"},
                    {"name": "Gulf Coast", "lat": 30.5, "lon": -88.0, "area": 720, "main_crop": "Peanuts"},
                    {"name": "Piedmont Region", "lat": 33.2, "lon": -85.8, "area": 680, "main_crop": "Wheat"}
                ]
                
                # Create DataFrame
                region_df = pd.DataFrame(region_data)
                
                # Create map visualization
                fig = px.scatter_mapbox(
                    region_df,
                    lat="lat",
                    lon="lon",
                    size="area",
                    color="main_crop",
                    hover_name="name",
                    hover_data=["area", "main_crop"],
                    zoom=6,
                    center={"lat": 32.8, "lon": -86.8},
                    title="Alabama Agricultural Regions"
                )
                
                fig.update_layout(mapbox_style="open-street-map", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading crop detection module: {str(e)}")
            st.info("Please check that the GeoAI library is properly installed and configured.")

    # Footer
    st.markdown("---")
    st.markdown("### 🛰️ Powered by NASA GeoAI Technology | Real-time Building Footprint & Crop Detection")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
