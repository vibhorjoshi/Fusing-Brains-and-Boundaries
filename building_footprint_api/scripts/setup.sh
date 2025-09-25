#!/bin/bash

# Create necessary directories
mkdir -p logs data temp models

# Copy example .env file if .env doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
    echo "Please edit .env file with your settings"
fi

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "Created virtual environment"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup completed successfully"
echo "Run 'source venv/bin/activate' to activate the virtual environment"
echo "Run 'uvicorn main:app --reload' to start the API server"