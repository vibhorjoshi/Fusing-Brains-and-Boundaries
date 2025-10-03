#!/bin/bash

# Update frontend requirements script for CI/CD
set -e

# Navigate to the frontend directory
cd "$(dirname "$0")"

# Install frontend dependencies
echo "Installing frontend dependencies..."
pip install -r requirements.txt

# Install npm packages if package.json exists
if [ -f "package.json" ]; then
  echo "Installing npm packages..."
  npm ci
  
  # Build frontend if build script exists
  if grep -q "\"build\"" package.json; then
    echo "Building frontend..."
    npm run build
  fi
else
  echo "No package.json found, skipping npm install."
fi

echo "Frontend setup completed successfully!"