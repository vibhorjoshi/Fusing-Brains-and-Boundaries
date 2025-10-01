import streamlit as st
import time
from datetime import datetime

st.set_page_config(page_title="GeoAI Live Pipeline", page_icon="", layout="wide")

st.title(" GeoAI Live Automation Pipeline")
st.markdown("### Real-time Building Footprint Detection for Alabama Cities")

# Live metrics dashboard
st.subheader(" Live Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Buildings Processed", "1,247", "+12")

with col2:
    st.metric("Detection Accuracy", "94.2%", "+0.1%")

with col3:
    st.metric("Active Processing Jobs", "3", " 1")

with col4:
    st.metric("Average Processing Time", "2.3s", "-0.2s")

# Processing pipeline visualization
st.subheader(" Processing Pipeline Status")

stages = ["Data Input", "Preprocessing", "AI Detection", "Post-processing", "Results Output"]
stage_icons = ["", "", "", "", ""]

cols = st.columns(len(stages))
for i, (stage, icon) in enumerate(zip(stages, stage_icons)):
    with cols[i]:
        st.write(f"{icon} {stage}")

# Active jobs monitoring
st.subheader(" Active Processing Jobs")

jobs_data = [
    {"city": "Birmingham", "status": "Processing", "progress": 65, "eta": "2 min"},
    {"city": "Montgomery", "status": "Queued", "progress": 0, "eta": "15 min"},
    {"city": "Mobile", "status": "Processing", "progress": 30, "eta": "8 min"},
    {"city": "Huntsville", "status": "Completed", "progress": 100, "eta": "Done"},
    {"city": "Tuscaloosa", "status": "Processing", "progress": 45, "eta": "5 min"}
]

for job in jobs_data:
    col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 1])
    
    with col1:
        st.write(f"**{job['city']}**")
    
    with col2:
        status_icon = "" if job["status"] == "Processing" else "" if job["status"] == "Queued" else ""
        st.write(f"{status_icon} {job['status']}")
    
    with col3:
        st.progress(job["progress"] / 100)
    
    with col4:
        st.write(f"{job['progress']}%")
    
    with col5:
        st.write(job["eta"])

# Alabama cities overview
st.subheader(" Alabama Cities Coverage")

cities_info = [
    {"name": "Birmingham", "buildings": 156421, "accuracy": 91.2},
    {"name": "Montgomery", "buildings": 98742, "accuracy": 89.7},
    {"name": "Mobile", "buildings": 87634, "accuracy": 88.4},
    {"name": "Huntsville", "buildings": 124563, "accuracy": 92.3},
    {"name": "Tuscaloosa", "buildings": 65432, "accuracy": 90.1}
]

st.dataframe(cities_info, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("###  Powered by NASA GeoAI Technology | Real-time Building Footprint Detection")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Auto-refresh every 5 seconds
time.sleep(5)
st.rerun()
