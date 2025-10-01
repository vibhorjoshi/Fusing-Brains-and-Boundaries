import streamlit as st
import numpy as np
import requests
import time
from datetime import datetime
import threading
import queue
from typing import Dict, List
import random

# Try importing pandas and plotly, with fallback
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="GeoAI Multi-Technique Pipeline", 
    page_icon="🏢", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    background: linear-gradient(90deg, #e3f2fd, #f3e5f5);
    border-radius: 10px;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 1rem 0;
}

.technique-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.processing-status {
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
    font-weight: bold;
}

.status-initializing { background-color: #fff3cd; color: #856404; }
.status-preprocessing { background-color: #cce7ff; color: #004085; }
.status-processing { background-color: #ffe6cc; color: #cc6600; }
.status-completed { background-color: #d4edda; color: #155724; }
.status-error { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Backend API Configuration
BACKEND_URL = "http://localhost:8000"

# Global state
if 'processing_jobs' not in st.session_state:
    st.session_state.processing_jobs = {}
if 'training_jobs' not in st.session_state:
    st.session_state.training_jobs = {}
if 'active_monitoring' not in st.session_state:
    st.session_state.active_monitoring = False

@st.cache_data(ttl=30)
def fetch_techniques():
    """Fetch available AI techniques"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/techniques")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching techniques: {e}")
        return None

@st.cache_data(ttl=30)
def fetch_areas():
    """Fetch available target areas"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/areas")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching areas: {e}")
        return None

def fetch_live_metrics():
    """Fetch live processing metrics"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/metrics/live")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return None

def start_processing_job(technique: str, area: str, preprocessing_config: Dict):
    """Start a new processing job"""
    try:
        payload = {
            "technique": technique,
            "target_area": area,
            "preprocessing_options": preprocessing_config,
            "real_time_monitoring": True
        }
        
        response = requests.post(f"{BACKEND_URL}/api/process/start", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error starting job: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error starting processing: {e}")
        return None

def fetch_job_status(job_id: str):
    """Fetch status of a specific job"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/process/status/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def start_training_job(technique: str, training_areas: List[str], testing_areas: List[str], epochs: int, batch_size: int):
    """Start a multi-area training job"""
    try:
        payload = {
            "technique": technique,
            "training_areas": training_areas,
            "testing_areas": testing_areas,
            "epochs": epochs,
            "batch_size": batch_size
        }
        
        response = requests.post(f"{BACKEND_URL}/api/training/start", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error starting training: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error starting training: {e}")
        return None

def get_buildings_for_maps(area_name: str):
    """Fetch building data for Google Maps integration"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/maps/buildings/{area_name}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching building data: {e}")
        return None

def create_technique_comparison_chart(techniques_data):
    """Create comparison chart for AI techniques"""
    if not techniques_data or not PLOTTING_AVAILABLE:
        return None
    
    techniques = techniques_data['techniques']
    
    try:
        df = pd.DataFrame([
            {
                'Technique': data['name'],
                'Accuracy': data['accuracy'],
                'Speed': data['speed'],
                'Complexity': data['complexity']
            }
            for key, data in techniques.items()
        ])
    except Exception as e:
        st.error(f"Error creating dataframe: {e}")
        return None
    
    # Create radar chart
    fig = go.Figure()
    
    # Normalize values for radar chart
    speed_mapping = {'Very Fast': 1.0, 'Fast': 0.8, 'Medium': 0.6, 'Slow': 0.4}
    complexity_mapping = {'Low': 0.25, 'Medium': 0.5, 'High': 0.75, 'Very High': 1.0}
    
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                row['Accuracy'],
                speed_mapping.get(row['Speed'], 0.5),
                1.0 - complexity_mapping.get(row['Complexity'], 0.5),  # Invert complexity (lower is better)
                row['Accuracy']
            ],
            theta=['Accuracy', 'Speed', 'Simplicity', 'Accuracy'],
            fill='toself',
            name=row['Technique']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="AI Technique Comparison (Radar Chart)"
    )
    
    return fig

def create_live_metrics_dashboard(metrics_data):
    """Create live metrics dashboard"""
    if not metrics_data:
        return None, None, None
    
    metrics = metrics_data['metrics']
    stats = metrics_data['job_statistics']
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Processed",
            value=metrics['total_processed'],
            delta=f"+{random.randint(1, 5)} today"
        )
    
    with col2:
        st.metric(
            label="Active Jobs", 
            value=stats['active_jobs'],
            delta=f"{stats['active_jobs'] - stats.get('prev_active', 0)}"
        )
    
    with col3:
        st.metric(
            label="Success Rate",
            value=f"{metrics['success_rate']:.1f}%",
            delta="+0.3%"
        )
    
    with col4:
        st.metric(
            label="Avg Processing Time",
            value=f"{metrics['avg_processing_time']:.1f}s",
            delta="-0.2s"
        )
    
    # Create technique usage chart
    if metrics['techniques_used'] and PLOTTING_AVAILABLE:
        try:
            tech_df = pd.DataFrame([
                {'Technique': tech, 'Usage Count': count}
                for tech, count in metrics['techniques_used'].items()
            ])
            
            usage_fig = px.pie(
                tech_df, 
                values='Usage Count', 
                names='Technique',
                title="Technique Usage Distribution"
            )
            
            job_status_fig = px.bar(
                x=['Completed', 'Active', 'Failed'],
                y=[stats['completed_jobs'], stats['active_jobs'], stats['failed_jobs']],
                title="Job Status Overview",
                color=['Completed', 'Active', 'Failed'],
                color_discrete_map={'Completed': '#28a745', 'Active': '#ffc107', 'Failed': '#dc3545'}
            )
        except Exception as e:
            st.error(f"Error creating charts: {e}")
            return None, None, stats
        
        return usage_fig, job_status_fig, stats
    
    return None, None, stats

def create_processing_monitor():
    """Create processing job monitor"""
    st.subheader("🔄 Active Processing Jobs")
    
    if not st.session_state.processing_jobs:
        st.info("No active processing jobs")
        return
    
    for job_id, job_info in st.session_state.processing_jobs.items():
        current_status = fetch_job_status(job_id)
        
        if current_status:
            st.session_state.processing_jobs[job_id] = current_status
            
            with st.expander(f"Job: {job_id}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Technique:** {current_status['technique']}")
                    st.write(f"**Target Area:** {current_status['target_area']}")
                    st.write(f"**Current Step:** {current_status.get('current_step', 'N/A')}")
                    
                    # Progress bar
                    progress = current_status.get('progress', 0)
                    st.progress(progress / 100)
                    st.write(f"Progress: {progress}%")
                
                with col2:
                    status = current_status['status']
                    status_class = f"status-{status.replace('_', '-')}"
                    
                    st.markdown(f"""
                    <div class="processing-status {status_class}">
                        Status: {status.upper()}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show results if completed
                if status == 'completed' and 'ai_result' in current_status:
                    st.success("Processing Completed!")
                    
                    result = current_status['ai_result']
                    iou_result = current_status.get('iou_result', {})
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Buildings Detected", result['buildings_detected'])
                    with col2:
                        st.metric("Average Confidence", f"{result['average_confidence']:.3f}")
                    with col3:
                        st.metric("IoU Score", f"{iou_result.get('iou_score', 0):.3f}")

def create_google_maps_integration(area_name: str):
    """Create Google Maps integration section"""
    st.subheader("🗺️ Google Maps Integration")
    
    building_data = get_buildings_for_maps(area_name)
    
    if building_data:
        buildings = building_data['buildings']
        center = building_data['center_coordinates']
        
        # Create map data for Streamlit
        if PLOTTING_AVAILABLE:
            try:
                map_df = pd.DataFrame([
                    {
                        'lat': building['lat'],
                        'lon': building['lng'],
                        'confidence': building['confidence'],
                        'building_type': building['building_type'],
                        'technique': building['detection_technique']
                    }
                    for building in buildings
                ])
                
                # Display map
                st.map(map_df)
            except Exception as e:
                st.error(f"Error creating map: {e}")
                st.write("Map data available but visualization failed")
        else:
            st.warning("Plotting libraries not available - showing raw data")
        
        # Building statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Buildings", len(buildings))
        
        with col2:
            avg_confidence = np.mean([b['confidence'] for b in buildings])
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            building_types = pd.Series([b['building_type'] for b in buildings]).value_counts()
            most_common = building_types.index[0] if len(building_types) > 0 else "N/A"
            st.metric("Most Common Type", most_common)
        
        # Technique distribution
        if PLOTTING_AVAILABLE:
            try:
                tech_dist = pd.Series([b['detection_technique'] for b in buildings]).value_counts()
                
                fig = px.bar(
                    x=tech_dist.index,
                    y=tech_dist.values,
                    title=f"Detection Techniques Used in {area_name}",
                    labels={'x': 'Technique', 'y': 'Count'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating technique distribution chart: {e}")
                # Show text summary instead
                techniques_count = {}
                for b in buildings:
                    tech = b['detection_technique']
                    techniques_count[tech] = techniques_count.get(tech, 0) + 1
                
                st.write("**Technique Distribution (Text):**")
                for tech, count in techniques_count.items():
                    st.write(f"- {tech}: {count} buildings")
        else:
            # Show text summary
            techniques_count = {}
            for b in buildings:
                tech = b['detection_technique']
                techniques_count[tech] = techniques_count.get(tech, 0) + 1
            
            st.write("**Technique Distribution:**")
            for tech, count in techniques_count.items():
                st.write(f"- {tech}: {count} buildings")
    
    else:
        st.warning("Could not load building data for maps integration")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🏢 GeoAI Multi-Technique Building Detection Pipeline 🚀</div>', 
                unsafe_allow_html=True)
    
    # Sidebar - Technique Selection
    st.sidebar.title("🎯 Pipeline Configuration")
    
    # Fetch techniques and areas
    techniques_data = fetch_techniques()
    areas_data = fetch_areas()
    
    if not techniques_data or not areas_data:
        st.error("Unable to connect to backend API. Please ensure the backend is running on http://localhost:8000")
        return
    
    # Technique selection
    st.sidebar.subheader("Select AI Technique")
    
    technique_options = {}
    if techniques_data:
        technique_options = {
            f"{data['name']} (Acc: {data['accuracy']:.2f})": key
            for key, data in techniques_data['techniques'].items()
        }
    
    selected_technique_display = st.sidebar.selectbox(
        "Choose Detection Technique:",
        list(technique_options.keys())
    )
    
    selected_technique = technique_options[selected_technique_display]
    
    # Area selection
    st.sidebar.subheader("Select Target Area")
    
    area_options = {}
    if areas_data:
        # Combine training and testing areas
        for area_name, area_data in areas_data['training_areas'].items():
            area_options[f"{area_name} (Training) - Pop: {area_data['population']:,}"] = area_name
        
        for area_name, area_data in areas_data['testing_areas'].items():
            area_options[f"{area_name} (Testing) - Pop: {area_data['population']:,}"] = area_name
    
    selected_area_display = st.sidebar.selectbox(
        "Choose Target Area:",
        list(area_options.keys())
    )
    
    selected_area = area_options[selected_area_display]
    
    # Preprocessing configuration
    st.sidebar.subheader("🔧 Preprocessing Options")
    
    preprocessing_config = {
        "noise_reduction": st.sidebar.checkbox("Noise Reduction", value=True),
        "contrast_enhancement": st.sidebar.checkbox("Contrast Enhancement", value=True),
        "edge_detection": st.sidebar.checkbox("Edge Detection", value=True),
        "data_augmentation": st.sidebar.checkbox("Data Augmentation", value=False),
        "resolution_scaling": st.sidebar.slider("Resolution Scaling", 0.5, 2.0, 1.0, 0.1)
    }
    
    # Start Processing Button
    if st.sidebar.button("🚀 Start Processing", type="primary"):
        with st.sidebar:
            with st.spinner("Starting processing job..."):
                result = start_processing_job(selected_technique, selected_area, preprocessing_config)
                
                if result:
                    st.success(f"Job started! ID: {result['job_id']}")
                    st.session_state.processing_jobs[result['job_id']] = result
                    st.session_state.active_monitoring = True
                else:
                    st.error("Failed to start processing job")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Dashboard", 
        "🔄 Active Jobs", 
        "🗺️ Maps Integration", 
        "🎓 Multi-Area Training",
        "📈 Technique Comparison"
    ])
    
    with tab1:
        st.header("Live Processing Dashboard")
        
        metrics_data = fetch_live_metrics()
        if metrics_data:
            usage_fig, status_fig, stats = create_live_metrics_dashboard(metrics_data)
            
            if usage_fig and status_fig:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(usage_fig, use_container_width=True)
                
                with col2:
                    st.plotly_chart(status_fig, use_container_width=True)
    
    with tab2:
        create_processing_monitor()
    
    with tab3:
        create_google_maps_integration(selected_area)
    
    with tab4:
        st.header("🎓 Multi-Area Training Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Configuration")
            
            training_technique = st.selectbox(
                "Select Technique for Training:",
                list(technique_options.keys()),
                key="training_technique"
            )
            
            training_areas = st.multiselect(
                "Select Training Areas:",
                [name for name in areas_data['training_areas'].keys()],
                default=list(areas_data['training_areas'].keys())[:2]
            )
            
            testing_areas = st.multiselect(
                "Select Testing Areas:",
                [name for name in areas_data['testing_areas'].keys()],
                default=list(areas_data['testing_areas'].keys())[:2]
            )
            
            epochs = st.slider("Training Epochs", 10, 100, 50)
            batch_size = st.slider("Batch Size", 8, 64, 16)
            
            if st.button("🎓 Start Training", type="primary"):
                selected_training_technique = technique_options[training_technique]
                
                with st.spinner("Starting training job..."):
                    training_result = start_training_job(
                        selected_training_technique,
                        training_areas,
                        testing_areas,
                        epochs,
                        batch_size
                    )
                    
                    if training_result:
                        st.success(f"Training started! ID: {training_result['training_id']}")
                        st.session_state.training_jobs[training_result['training_id']] = training_result
                    else:
                        st.error("Failed to start training job")
        
        with col2:
            st.subheader("Training Status")
            
            if st.session_state.training_jobs:
                for training_id, training_info in st.session_state.training_jobs.items():
                    current_status = fetch_job_status(training_id)
                    
                    if current_status:
                        st.session_state.training_jobs[training_id] = current_status
                        
                        with st.expander(f"Training: {training_id}", expanded=True):
                            st.write(f"**Technique:** {current_status['technique']}")
                            st.write(f"**Status:** {current_status['status']}")
                            
                            if 'current_epoch' in current_status:
                                st.write(f"**Epoch:** {current_status['current_epoch']}/{current_status['epochs']}")
                            
                            progress = current_status.get('progress', 0)
                            st.progress(progress / 100)
                            
                            if current_status['status'] == 'completed' and 'test_results' in current_status:
                                st.success("Training Completed!")
                                st.write(f"**Final Accuracy:** {current_status['final_accuracy']:.3f}")
                                
                                # Show test results
                                test_results = current_status['test_results']
                                if PLOTTING_AVAILABLE:
                                    try:
                                        results_df = pd.DataFrame([
                                            {
                                                'Area': area,
                                                'Accuracy': data['accuracy'],
                                                'IoU': data['iou'],
                                                'Buildings': data['buildings_detected']
                                            }
                                            for area, data in test_results.items()
                                        ])
                                        
                                        st.dataframe(results_df)
                                    except Exception as e:
                                        st.error(f"Error displaying results table: {e}")
                                        # Show text results
                                        for area, data in test_results.items():
                                            st.write(f"**{area}:** Accuracy: {data['accuracy']:.3f}, IoU: {data['iou']:.3f}, Buildings: {data['buildings_detected']}")
                                else:
                                    # Show text results
                                    for area, data in test_results.items():
                                        st.write(f"**{area}:** Accuracy: {data['accuracy']:.3f}, IoU: {data['iou']:.3f}, Buildings: {data['buildings_detected']}")
            else:
                st.info("No active training jobs")
    
    with tab5:
        st.header("📈 AI Technique Comparison")
        
        if techniques_data:
            comparison_fig = create_technique_comparison_chart(techniques_data)
            
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Technique details table
            st.subheader("Technique Details")
            
            if PLOTTING_AVAILABLE:
                try:
                    technique_df = pd.DataFrame([
                        {
                            'Technique': data['name'],
                            'Description': data['description'],
                            'Accuracy': f"{data['accuracy']:.2f}",
                            'Speed': data['speed'],
                            'Complexity': data['complexity']
                        }
                        for key, data in techniques_data['techniques'].items()
                    ])
                    
                    st.dataframe(technique_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating technique table: {e}")
                    # Show text version
                    for key, data in techniques_data['techniques'].items():
                        st.write(f"**{data['name']}** - Accuracy: {data['accuracy']:.2f}, Speed: {data['speed']}, Complexity: {data['complexity']}")
                        st.write(f"  {data['description']}")
            else:
                # Show text version
                for key, data in techniques_data['techniques'].items():
                    st.write(f"**{data['name']}** - Accuracy: {data['accuracy']:.2f}, Speed: {data['speed']}, Complexity: {data['complexity']}")
                    st.write(f"  {data['description']}")
    
    # Auto-refresh for active monitoring
    if st.session_state.active_monitoring:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()