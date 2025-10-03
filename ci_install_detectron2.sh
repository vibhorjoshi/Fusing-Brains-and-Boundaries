#!/bin/bash
# Install detectron2 for Linux CI/CD pipeline
# This script should be run after installing the main requirements

echo "Installing detectron2 from source..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo "Installing additional dependencies..."
python -m pip install -r additional_requirements.txt

echo "Installation complete."