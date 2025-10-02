#!/bin/bash
# Backend installation script for CI
set -e

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install dependencies
pip install -r backend/requirements.txt

# Verify installation
echo "Backend dependencies installed successfully"
python -c "import numpy; import pandas; import fastapi; print('Key backend packages imported successfully')"