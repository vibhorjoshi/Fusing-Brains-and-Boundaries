#!/usr/bin/env python
"""
GeoAI Project Dependency Setup
This script installs all dependencies correctly for various environments.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

def run_command(cmd, description=None):
    """Run a command and return the result"""
    if description:
        print(f"\n{description}...")
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Success!")
        if result.stdout.strip():
            print(result.stdout.strip())
    else:
        print(f"Error (code {result.returncode}):")
        print(result.stderr.strip())
    
    return result.returncode == 0

def is_conda_available():
    """Check if conda is available"""
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def install_dependencies_pip(ci_mode=False):
    """Install dependencies using pip"""
    req_file = "requirements_ci.txt" if ci_mode else "requirements.txt"
    success = run_command(f"pip install -r {req_file}", f"Installing dependencies from {req_file}")
    
    if success:
        # Try to install detectron2
        try:
            print("\nAttempting to install detectron2 from source...")
            subprocess.run(
                ["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print("Detectron2 installed successfully!")
        except subprocess.SubprocessError as e:
            print(f"Could not install detectron2: {str(e)}")
            print("Continuing without detectron2. Some features may not work.")
    
    return success

def install_dependencies_conda():
    """Install dependencies using conda"""
    return run_command("conda env update -f environment.yml", "Creating/updating conda environment")

def setup_directories():
    """Set up necessary directories"""
    directories = [
        "data",
        "data/usa",
        "outputs",
        "outputs/runner_logs",
        "outputs/usa_metrics",
        "frontend"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    parser = argparse.ArgumentParser(description='GeoAI Project Dependency Setup')
    parser.add_argument('--ci', action='store_true', help='Use CI/CD compatible dependencies')
    parser.add_argument('--conda', action='store_true', help='Force conda installation')
    parser.add_argument('--pip', action='store_true', help='Force pip installation')
    args = parser.parse_args()
    
    print("=" * 60)
    print("GeoAI Project Dependency Setup")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Set up directories
    setup_directories()
    
    # Install dependencies
    if args.conda or (is_conda_available() and not args.pip):
        print("\nUsing conda to install dependencies...")
        success = install_dependencies_conda()
    else:
        print("\nUsing pip to install dependencies...")
        success = install_dependencies_pip(args.ci)
    
    # Final status message
    if success:
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nYou can now run the monitoring system with:")
        if platform.system() == "Windows":
            print("  python run_with_monitoring_fixed.py")
        else:
            print("  python3 run_with_monitoring_fixed.py")
    else:
        print("\n" + "=" * 60)
        print("Setup encountered some errors.")
        print("Please check the output above and try to resolve any issues.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()