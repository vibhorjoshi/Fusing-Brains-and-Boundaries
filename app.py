# Fix matplotlib and fontconfig permissions for containerized environments
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['FONTCONFIG_PATH'] = '/tmp/fontconfig'
os.environ['FONTCONFIG_FILE'] = '/tmp/fontconfig/fonts.conf'

import gradio as gr
import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.config import Config
    from src.inference import BuildingFootprintExtractor
    CONFIG_LOADED = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    CONFIG_LOADED = False

# Initialize configuration and create directories
if CONFIG_LOADED:
    try:
        Config.create_directories()
        print("✅ Directories created successfully")
    except Exception as e:
        print(f"⚠️ Could not create directories: {e}")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Initialize the pipeline
extractor = None
if CONFIG_LOADED:
    try:
        extractor = BuildingFootprintExtractor(config=Config, device=device)
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"⚠️ Could not initialize full pipeline: {e}")
        print("📝 Traceback:", traceback.format_exc())
        extractor = None

def process_image(image, method="RL Fusion"):
    """Process an image and extract building footprints."""
    if not CONFIG_LOADED:
        return None, "❌ Configuration not loaded. Check server logs."
    
    if extractor is None:
        return None, "⚠️ Pipeline not initialized. Running in demo mode."
    
    if image is None:
        return None, "❌ Please upload an image first."
    
    try:
        # Process the image
        result = extractor.extract(image, method=method)
        stats = f"""✅ Processing Complete!

Method: {method}
Device: {device}
Status: {result.get('status', 'Success')}

Image Shape: {result.get('input_shape', 'Unknown')}
Buildings Detected: {result.get('buildings_count', 0)}
Processing Time: {result.get('processing_time', 'N/A')}

{result.get('additional_info', '')}"""
        
        return result["visualization"], stats
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}\n\n📝 Traceback:\n{traceback.format_exc()}"
        return None, error_msg

def get_example_images():
    """Get example images if available."""
    examples_dir = Path("data/test_images")
    if examples_dir.exists():
        return [str(p) for p in examples_dir.glob("*.{jpg,jpeg,png,tif}")]
    return []

# Create Gradio interface
with gr.Blocks(
    title="🏗️ Building Footprint Extraction",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    """
) as demo:
    gr.Markdown("""
    # 🏗️ Building Footprint Extraction
    
    GPU-accelerated building footprint extraction using **Mask R-CNN** and **adaptive fusion** with reinforcement learning.
    
    📍 Upload a satellite image to detect and extract building footprints automatically.
    """)
    
    # System status
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"""
            ### 🔧 System Status
            - **Device**: {device}
            - **Config**: {'✅ Loaded' if CONFIG_LOADED else '❌ Failed'}
            - **Pipeline**: {'✅ Ready' if extractor else '⚠️ Demo Mode'}
            """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            input_image = gr.Image(
                label="Satellite Image", 
                type="numpy",
                height=400
            )
            
            method = gr.Dropdown(
                choices=["Mask R-CNN", "RT", "RR", "FER", "RL Fusion"],
                value="RL Fusion",
                label="🎯 Extraction Method",
                info="Select the building extraction algorithm"
            )
            
            submit_btn = gr.Button(
                "🚀 Extract Buildings", 
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Results")
            output_image = gr.Image(
                label="Extracted Buildings",
                height=400
            )
            output_stats = gr.Textbox(
                label="📊 Processing Statistics", 
                lines=10,
                max_lines=15
            )
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, method],
        outputs=[output_image, output_stats]
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## 📖 How to Use
            
            1. **Upload Image**: Click on the image box and upload a satellite/aerial image
            2. **Select Method**: Choose from the available extraction methods
            3. **Extract**: Click "Extract Buildings" to process the image
            4. **View Results**: See the detected buildings highlighted on the output image
            
            ## 🎯 Methods Available
            
            - **🤖 Mask R-CNN**: Base deep learning object detector
            - **🔧 RT**: Regular Topology regularization 
            - **📐 RR**: Regular Rectangle regularization
            - **🎨 FER**: Feature Edge Regularization
            - **🧠 RL Fusion**: Adaptive fusion using reinforcement learning ⭐ **(Recommended)**
            
            ## 💡 Tips
            
            - Use high-resolution satellite images for best results
            - **RL Fusion** typically provides the most accurate results
            - Processing time depends on image size and selected method
            - GPU acceleration is automatically used when available
            """)
    
    # Footer
    gr.Markdown("""
    ---
    🚀 **Fusing Brains and Boundaries** - Advanced Building Footprint Extraction Pipeline
    
    🔬 Research project combining deep learning, regularization techniques, and reinforcement learning.
    """)

if __name__ == "__main__":
    print("🚀 Starting Building Footprint Extraction App...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_error=True
    )