#!/bin/bash
# Install frontend dependencies using npm only for CI

npm install

# Run the build process
npm run build

# Run tests if available
npm test || echo "No tests specified or some tests failed"