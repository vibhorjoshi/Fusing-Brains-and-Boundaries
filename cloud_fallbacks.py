"""
Fallback modules for cloud deployment
"""
# This file serves as a fallback when src modules can't be directly imported

# Minimal implementation of key classes needed for Streamlit app
class PipelineStage:
    def __init__(self, name, input_data, output_data, processing_time, metrics):
        self.name = name
        self.input_data = input_data
        self.output_data = output_data
        self.processing_time = processing_time
        self.metrics = metrics

class FewShotRLPipeline:
    def __init__(self, *args, **kwargs):
        pass
    
    def run(self, *args, **kwargs):
        return {"status": "demo_mode", "message": "Running in demo mode"}

class OpenSourceGeoAI:
    def __init__(self, *args, **kwargs):
        pass
    
    def process(self, *args, **kwargs):
        return {"status": "demo_mode", "message": "Running in demo mode"}

class LiveAutomationPipeline:
    def __init__(self, *args, **kwargs):
        pass
    
    def run(self, *args, **kwargs):
        return {"status": "demo_mode", "message": "Running in demo mode"}

class LiveResultsVisualization:
    def __init__(self, *args, **kwargs):
        pass
    
    def display_live_results_section(self, automation_results=None):
        """Display results section in demo mode"""
        import streamlit as st
        st.info("🔄 Running in demo mode - Live visualization not available")
        st.markdown("**Demo Mode:** Real-time visualization features are disabled.")
        return {"status": "demo_mode", "message": "Running in demo mode"}
    
    def visualize(self, *args, **kwargs):
        return {"status": "demo_mode", "message": "Running in demo mode"}

class Config:
    """Minimal config class with default values"""
    def __init__(self):
        self.DEFAULT_STATE = "demo"
        self.PATCH_SIZE = 256
        self.NUM_WORKERS = 2
        self.BATCH_SIZE = 4

def run_automation_demo(*args, **kwargs):
    """Demo function for automation pipeline"""
    return {"status": "demo_mode", "message": "Running in demo mode"}