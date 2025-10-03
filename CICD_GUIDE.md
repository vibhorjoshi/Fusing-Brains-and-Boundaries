# CI/CD Pipeline Guide

## Overview

This guide explains how the Continuous Integration and Continuous Deployment (CI/CD) pipeline works for the Real USA Agricultural Detection System. The pipeline is designed to automate testing, building, and deployment processes, ensuring consistent quality and seamless delivery of the application.

## Pipeline Architecture

Our CI/CD pipeline follows the original architecture of the system:

```
Preprocessing → MaskRCNN → RR RT FER → Adaptive Fusion → Post-processing
```

With the following key components:

1. **Backend API**: FastAPI service providing access to the GeoAI engine
2. **Frontend**: Web interface for visualizing building footprints and agricultural areas
3. **Streamlit Dashboard**: Interactive analytics dashboard for agricultural detection
4. **Redis**: In-memory data store for caching and message passing
5. **Nginx**: Reverse proxy for production deployment

## Workflow Steps

The CI/CD pipeline consists of the following steps:

### 1. Validate Configuration Files

Ensures all required configuration files are present in the repository:
- Requirements files
- Docker configuration files
- Deployment scripts

### 2. Code Quality Checks

Runs automated tools to ensure code quality:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **Bandit**: Security analysis

### 3. Test Python Code

Runs automated tests for the Python codebase:
- Unit tests
- Integration tests
- Coverage reporting

### 4. Test Streamlit App

Validates the Streamlit dashboard:
- Syntax checking
- Module imports
- Basic functionality

### 5. Build Docker Images

Builds Docker images for all components:
- Backend API
- Frontend
- Streamlit Dashboard
- GeoAI Engine

### 6. Integration Tests

Tests the complete system integration:
- Start all services with Docker Compose
- Test API endpoints
- Verify database connections
- Check frontend connectivity

### 7. Deployment

Deploys the application to the appropriate environment:
- Development: For feature branch pushes
- Staging: For `develop` branch pushes
- Production: For `main` branch pushes or manual triggers

## Usage

### Triggering the Pipeline

The pipeline is triggered automatically on:
- **Push** to `main` or `develop` branches
- **Pull requests** targeting `main` or `develop` branches
- **Manual trigger** via GitHub Actions interface

### Manual Deployment

You can also deploy the system manually using the provided deployment script:

```bash
python deploy.py --env production
```

Options:
- `--env` or `-e`: Environment to deploy to (`development`, `staging`, or `production`)
- `--skip-checks` or `-s`: Skip dependency checks
- `--no-pull` or `-n`: Skip pulling latest images

## Environment Variables

The pipeline uses the following environment variables:

| Variable | Description |
|----------|-------------|
| `REGISTRY` | Container registry URL |
| `IMAGE_NAME_PREFIX` | Prefix for Docker image names |
| `PYTHON_VERSION` | Python version for testing |
| `NODE_VERSION` | Node.js version for frontend tests |

## Secrets

The following secrets are required in your GitHub repository:

| Secret | Description |
|--------|-------------|
| `GITHUB_TOKEN` | Automatically provided by GitHub |
| `AWS_ACCESS_KEY_ID` | (Optional) AWS access key for deployment |
| `AWS_SECRET_ACCESS_KEY` | (Optional) AWS secret for deployment |

## Customizing the Pipeline

To customize the CI/CD pipeline:

1. Edit `.github/workflows/enhanced-ci-cd-pipeline.yml` to change workflow steps
2. Modify `deploy.py` to customize deployment logic
3. Update Docker Compose files to adjust service configuration

## Troubleshooting

### Common Issues

#### Failed Build

Check the build logs for specific errors. Common issues include:
- Missing dependencies
- Failed tests
- Docker build errors

#### Failed Deployment

If deployment fails:
1. Verify environment variables are correctly set
2. Check deployment logs
3. Ensure target environment is accessible

#### Integration Test Failures

If integration tests fail:
1. Check Docker container logs
2. Verify service connectivity
3. Validate API responses

## Best Practices

1. **Keep tests updated**: Maintain comprehensive test coverage
2. **Version your releases**: Use semantic versioning for releases
3. **Monitor deployments**: Check logs after deployment
4. **Document changes**: Include detailed commit messages

## Architecture Diagram

```
┌───────────┐       ┌───────────┐       ┌───────────┐
│           │       │           │       │           │
│   Build   ├───────┤   Test    ├───────┤  Deploy   │
│           │       │           │       │           │
└───────────┘       └───────────┘       └───────────┘
      │                   │                   │
      ▼                   ▼                   ▼
┌───────────┐       ┌───────────┐       ┌───────────┐
│  Backend  │       │   API     │       │Production │
│  Frontend │       │Integration│       │  Staging  │
│ Streamlit │       │   Tests   │       │    Dev    │
└───────────┘       └───────────┘       └───────────┘
```