# GeoAI CI/CD Pipeline Setup

This guide explains how to set up and manage the CI/CD pipeline for the GeoAI project.

## Dependencies Management

The project requires several complex dependencies, including `detectron2` which is not available on PyPI. Several approaches are provided to handle these dependencies:

### Option 1: Using the setup script

The easiest way to set up dependencies is to use the provided setup script:

```bash
# For regular development:
python setup_dependencies.py

# For CI/CD environments:
python setup_dependencies.py --ci
```

### Option 2: Manual installation

#### For standard environments:

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git
```

#### For CI/CD environments:

```bash
pip install -r requirements_ci.txt
# Then based on OS:
bash ci_install_detectron2.sh  # Linux/macOS
# OR
.\ci_install_detectron2.ps1    # Windows PowerShell
```

### Option 3: Using conda (recommended for complex dependencies)

```bash
conda env update -f environment.yml
conda activate geoai
```

## Common Issues and Solutions

### Detectron2 Installation Failures

If detectron2 fails to install, ensure you have the necessary build tools:

#### Linux:
```bash
apt-get update && apt-get install -y build-essential git
```

#### macOS:
```bash
brew install cmake
```

#### Windows:
```powershell
# Ensure you have Visual Studio Build Tools installed
pip install -U torch torchvision
```

### Python Version Compatibility

This project is tested with Python 3.9. Tensorflow and some other packages may have version conflicts with newer Python versions.

### Error: No matching distribution found for detectron2

This is expected as detectron2 is not on PyPI. Use the GitHub installation method instead:

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

## GitHub Actions CI/CD Setup

The provided `.github/workflows/ci_cd_pipeline.yml` file configures a CI/CD pipeline using GitHub Actions. This workflow:

1. Sets up a Python 3.9 environment
2. Installs system dependencies
3. Installs Python dependencies using the CI-compatible requirements
4. Installs detectron2 from GitHub
5. Runs linting checks and tests
6. Deploys the application (on main branch only)

## Docker Support

For consistent environments across development and production, consider using the provided Docker support:

```bash
docker build -t geoai:latest .
docker run -p 8502:8502 -p 8080:8080 -p 9090:9090 geoai:latest
```