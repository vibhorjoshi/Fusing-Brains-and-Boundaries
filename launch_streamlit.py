"""
Launcher script for the Streamlit GeoAI application
This script ensures proper environment setup before launching the Streamlit app

Usage:
    python launch_streamlit.py
"""

import os
import sys
import subprocess
import time
import platform

def get_python_executable():
    """Get the correct Python executable based on platform and environment"""
    # If in a virtual environment, use that Python
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.executable
    
    # Otherwise try to find Python
    if platform.system() == "Windows":
        try:
            # Try Python from PATH
            result = subprocess.check_output(["where", "python"], text=True)
            paths = result.strip().split('\n')
            if paths:
                return paths[0]
        except:
            # Fallbacks for Windows
            common_paths = [
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python311", "python.exe"),
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Python311", "python.exe"),
                "C:\\Python311\\python.exe",
                "python"  # Last resort
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
    else:
        # Linux/Mac paths
        try:
            return subprocess.check_output(["which", "python3"], text=True).strip()
        except:
            return "python3"  # Last resort
    
    # If nothing found, return default
    return "python"

def install_pip_if_needed(python_exe):
    """Make sure pip is installed"""
    try:
        subprocess.check_call([python_exe, "-m", "pip", "--version"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        return True
    except:
        print("📦 Pip not found. Installing pip...")
        try:
            subprocess.check_call([python_exe, "-m", "ensurepip", "--upgrade"])
            print("✅ Pip installed successfully")
            return True
        except:
            print("❌ Could not install pip automatically")
            print("Please install pip manually: https://pip.pypa.io/en/stable/installation/")
            return False

def check_requirements(python_exe):
    """Check if all required packages are installed using the specified Python"""
    print("🔍 Checking required packages...")
    
    # Define required packages
    required_packages = ["streamlit", "pandas", "numpy", "plotly", "requests", "pillow"]
    missing_packages = []
    
    for package in required_packages:
        try:
            # Try to import the package through Python
            cmd = f"import {package.lower()}; print('{package} ' + {package.lower()}.__version__)"
            result = subprocess.run(
                [python_exe, "-c", cmd],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                missing_packages.append(package)
            else:
                print(f"  ✓ {result.stdout.strip()}")
        except:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def install_requirements(python_exe):
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        # Upgrade pip first
        subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([python_exe, "-m", "pip", "install", "-r", "streamlit_requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def launch_streamlit(python_exe):
    """Launch the Streamlit app"""
    print("\n🚀 Launching Streamlit application...")
    try:
        # Set environment variables to configure Streamlit
        os.environ["STREAMLIT_SERVER_PORT"] = "8501"
        
        # Launch Streamlit app
        subprocess.call([python_exe, "-m", "streamlit", "run", "streamlit_app.py"])
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)

def check_src_module():
    """Check if src module is accessible"""
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import src
        return True
    except ImportError:
        return False

def create_mock_files_if_needed():
    """Create mock files if needed for demo purposes"""
    # Create mock GeoAI crop detection module if it doesn't exist
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    os.makedirs(src_dir, exist_ok=True)
    
    crop_detection_file = os.path.join(src_dir, "geoai_crop_detection.py")
    if not os.path.exists(crop_detection_file):
        with open(crop_detection_file, "w") as f:
            f.write("""
\"\"\"
Mock crop detection module for demo purposes
\"\"\"
import numpy as np
import random

def detect_agricultural_crops(image, region=None):
    \"\"\"Mock crop detection function\"\"\"
    # Generate mock crop detection results
    return {
        "crop_detections": [
            {"crop_type": "corn", "confidence": 0.92, "area_percentage": 45},
            {"crop_type": "soybean", "confidence": 0.87, "area_percentage": 30},
            {"crop_type": "wheat", "confidence": 0.89, "area_percentage": 25}
        ],
        "agricultural_area_percentage": 0.75,
        "visualization": None  # Would be an image in a real implementation
    }
""")
    
    # Create __init__.py if it doesn't exist
    init_file = os.path.join(src_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# GeoAI module initialization")

def main():
    """Main function"""
    print("🏗️ GeoAI Streamlit Application Launcher")
    print("=======================================")
    
    # Get the correct Python executable
    python_exe = get_python_executable()
    print(f"🐍 Using Python: {python_exe}")
    
    # Make sure pip is installed
    if not install_pip_if_needed(python_exe):
        sys.exit(1)
    
    # Check if requirements are installed
    if not check_requirements(python_exe):
        print("📝 Missing requirements. Installing now...")
        if not install_requirements(python_exe):
            print("❌ Could not install requirements. Please install them manually:")
            print("   pip install -r streamlit_requirements.txt")
            sys.exit(1)
    
    # Make sure src module is in path and create mock files if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    if not check_src_module():
        print("📂 Creating demo files for first-time setup...")
        create_mock_files_if_needed()
    
    # Launch Streamlit
    launch_streamlit(python_exe)

if __name__ == "__main__":
    main()