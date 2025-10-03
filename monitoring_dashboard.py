#!/usr/bin/env python
"""
GeoAI Monitoring Dashboard

This Streamlit dashboard provides monitoring and metrics for the GeoAI project.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="GeoAI Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Constants
OUTPUT_DIR = Path("outputs/runner_logs")
RESOURCE_FILE = OUTPUT_DIR / "resource_usage.json"
USA_METRICS_DIR = Path("outputs/usa_metrics")
USA_METRICS_FILE = USA_METRICS_DIR / "usa_metrics_processed.json"

# Helper functions
def load_resource_data():
    """Load resource usage data"""
    if not RESOURCE_FILE.exists():
        return None
    try:
        with open(RESOURCE_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading resource data: {str(e)}")
        return None

def load_usa_metrics():
    """Load USA metrics data"""
    if not USA_METRICS_FILE.exists():
        return None
    try:
        with open(USA_METRICS_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading USA metrics: {str(e)}")
        return None

def check_service_status():
    """Check if all services are running"""
    resource_data = load_resource_data()
    if not resource_data:
        return {"streamlit": "unknown", "frontend": "unknown", "api": "unknown"}
    
    status = {}
    for name, process_info in resource_data.get("processes", {}).items():
        status[name] = process_info.get("status", "unknown")
    
    return status

def load_log_file(log_file):
    """Load and return the last 100 lines from a log file"""
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                lines = f.readlines()
            return lines[-100:] if lines else []
        return []
    except Exception as e:
        st.error(f"Error reading log file {log_file}: {str(e)}")
        return []

# Title and description
st.title("GeoAI System Monitoring Dashboard")
st.markdown("""
This dashboard provides real-time monitoring of the GeoAI system, including:
- Service status and resource usage
- USA agricultural detection metrics
- Log monitoring and error tracking
""")

# Service Status Section
st.header("Service Status")
status = check_service_status()

col1, col2, col3 = st.columns(3)
with col1:
    streamlit_status = status.get("streamlit", "unknown")
    color = "green" if streamlit_status == "running" else "red"
    st.markdown(f"<h3 style='color: {color};'>Streamlit: {streamlit_status}</h3>", unsafe_allow_html=True)

with col2:
    frontend_status = status.get("frontend", "unknown")
    color = "green" if frontend_status == "running" else "red"
    st.markdown(f"<h3 style='color: {color};'>Frontend: {frontend_status}</h3>", unsafe_allow_html=True)

with col3:
    monitoring_status = status.get("monitoring", "unknown")
    color = "green" if monitoring_status == "running" else "red"
    st.markdown(f"<h3 style='color: {color};'>Monitoring: {monitoring_status}</h3>", unsafe_allow_html=True)

# Resource Usage Section
st.header("Resource Usage")
resource_data = load_resource_data()

if resource_data:
    system_data = resource_data.get("system", {})
    cpu_percent = system_data.get("cpu_percent", 0)
    memory_percent = system_data.get("memory_percent", 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        # Create CPU usage progress bar
        st.progress(cpu_percent / 100)
        
    with col2:
        st.metric("Memory Usage", f"{memory_percent:.1f}%")
        # Create memory usage progress bar
        st.progress(memory_percent / 100)
    
    # Process specific resource usage
    st.subheader("Process Resource Usage")
    process_data = []
    for name, process_info in resource_data.get("processes", {}).items():
        if process_info.get("status") == "running":
            process_data.append({
                "Process": name,
                "CPU %": process_info.get("cpu_percent", 0),
                "Memory %": process_info.get("memory_percent", 0),
                "Status": process_info.get("status", "unknown")
            })
    
    if process_data:
        st.dataframe(pd.DataFrame(process_data))
    else:
        st.info("No active processes detected")
else:
    st.warning("Resource usage data not available. Please run the monitoring system.")

# USA Metrics Section
st.header("USA Agricultural Detection Metrics")
usa_metrics = load_usa_metrics()

if usa_metrics:
    raw_metrics = usa_metrics.get("raw_metrics", {})
    aggregated = usa_metrics.get("aggregated", {})
    generated_at = usa_metrics.get("generated_at", "Unknown")
    
    st.markdown(f"**Last Updated:** {generated_at}")
    
    # Display overall metrics
    st.subheader("Overall Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision", f"{aggregated.get('overall_precision', 0):.3f}")
    with col2:
        st.metric("Recall", f"{aggregated.get('overall_recall', 0):.3f}")
    with col3:
        st.metric("F1 Score", f"{aggregated.get('overall_f1', 0):.3f}")
    
    # Regional metrics
    st.subheader("Regional Performance")
    
    # Prepare data for visualization
    regions = list(raw_metrics.keys())
    
    if regions:
        metrics_df = pd.DataFrame({
            "Region": regions,
            "Precision": [raw_metrics[region].get("precision", 0) for region in regions],
            "Recall": [raw_metrics[region].get("recall", 0) for region in regions],
            "F1 Score": [raw_metrics[region].get("f1_score", 0) for region in regions],
            "Processing Time": [raw_metrics[region].get("processing_time", 0) for region in regions],
            "Coverage": [raw_metrics[region].get("coverage", 0) for region in regions],
        })
        
        st.dataframe(metrics_df)
        
        # Plot metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up bar positions
        x = np.arange(len(regions))
        width = 0.25
        
        # Plot bars
        ax.bar(x - width, metrics_df["Precision"], width, label="Precision")
        ax.bar(x, metrics_df["Recall"], width, label="Recall")
        ax.bar(x + width, metrics_df["F1 Score"], width, label="F1 Score")
        
        # Add labels and legend
        ax.set_ylabel("Score")
        ax.set_title("Detection Quality by Region")
        ax.set_xticks(x)
        ax.set_xticklabels([r.capitalize() for r in regions])
        ax.legend()
        
        st.pyplot(fig)
        
        # Processing Time Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Region", y="Processing Time", data=metrics_df, ax=ax)
        ax.set_title("Processing Time by Region (seconds per km²)")
        st.pyplot(fig)
    else:
        st.info("No regional data available")
else:
    st.warning("USA metrics data not available. Run the monitoring system to generate metrics.")

# Log Monitoring Section
st.header("Log Monitoring")

log_tabs = st.tabs(["Runner Logs", "Streamlit Logs"])

with log_tabs[0]:
    runner_logs = load_log_file("geoai_runner.log")
    if runner_logs:
        st.code("".join(runner_logs), language="text")
    else:
        st.info("No runner logs available")

with log_tabs[1]:
    streamlit_logs = load_log_file("streamlit_backend.log")
    if streamlit_logs:
        st.code("".join(streamlit_logs), language="text")
    else:
        st.info("No Streamlit logs available")

# Error Analysis Section
st.header("Error Analysis")
error_keywords = ["error", "exception", "failed", "traceback", "critical"]

# Function to extract errors from logs
def extract_errors(log_lines):
    errors = []
    for i, line in enumerate(log_lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in error_keywords):
            errors.append(line)
    return errors

runner_errors = extract_errors(load_log_file("geoai_runner.log"))
streamlit_errors = extract_errors(load_log_file("streamlit_backend.log"))

if runner_errors or streamlit_errors:
    error_tabs = st.tabs(["Runner Errors", "Streamlit Errors"])
    
    with error_tabs[0]:
        if runner_errors:
            st.code("".join(runner_errors), language="text")
        else:
            st.success("No errors found in runner logs")
    
    with error_tabs[1]:
        if streamlit_errors:
            st.code("".join(streamlit_errors), language="text")
        else:
            st.success("No errors found in Streamlit logs")
else:
    st.success("No errors found in any log files")

# Add auto-refresh
st.sidebar.title("Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 30s)", value=True)

if auto_refresh:
    st.sidebar.info("Dashboard will refresh every 30 seconds")
    time_counter = st.sidebar.empty()
    refresh_button = st.sidebar.button("Refresh Now")
    
    if not refresh_button:
        # Add JavaScript for auto-refresh
        st.markdown(
            """
            <script>
                function refreshPage() {
                    setTimeout(function () {
                        location.reload();
                    }, 30000);
                }
                refreshPage();
            </script>
            """,
            unsafe_allow_html=True
        )
else:
    refresh_button = st.sidebar.button("Refresh Now")
    if refresh_button:
        st.experimental_rerun()