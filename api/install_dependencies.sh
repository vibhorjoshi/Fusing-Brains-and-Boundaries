#!/bin/bash
# This script installs dependencies and handles failures gracefully for Vercel deployment
set -e

echo "Installing dependencies from requirements.txt..."
# Try to install all requirements, but don't fail if some packages can't be installed
pip install -r ../vercel.requirements.txt || true

echo "Installation completed (some packages may have been skipped)."
exit 0  # Always exit with success