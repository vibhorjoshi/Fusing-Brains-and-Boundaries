#!/bin/bash

# Update backend requirements script for CI/CD
set -e

# Navigate to the backend directory
cd "$(dirname "$0")"

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r requirements.txt

# Run validation tests if they exist
if [ -d "tests" ]; then
  echo "Running backend validation tests..."
  pytest tests -v
else
  echo "No backend tests found, skipping validation."
fi

echo "Backend setup completed successfully!"