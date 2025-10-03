#!/bin/bash
# Netlify build script for handling Python dependencies gracefully

set -e

echo "Starting Netlify build process..."

# Install Python dependencies from custom requirements
echo "Installing Python dependencies..."
pip install -r netlify.requirements.txt || true
echo "Python dependencies installation completed (some packages may have been skipped)"

# Build the frontend
echo "Building Next.js frontend..."
cd frontend
npm install
npm run build

echo "Build process completed successfully!"
exit 0