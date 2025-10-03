# CI/CD Pipeline for GeoAI Research

This document provides information about the CI/CD pipeline implemented for the GeoAI research project.

## Pipeline Overview

Our CI/CD pipeline automates the following processes:

1. **Code Validation and Testing**
   - Configuration validation
   - Code quality checks (linting)
   - Unit and integration testing
   - Streamlit application testing

2. **Docker Build and Deployment**
   - Docker image building with caching
   - Docker deployment testing
   - Multi-environment deployment (dev, staging, prod)

3. **Documentation and Reporting**
   - Automatic API documentation generation
   - Performance testing and reporting
   - Build and deployment reporting

4. **Performance and Security**
   - Component benchmarking
   - Load testing
   - Cache optimization

## Running the Pipeline

### In GitHub Environment

If you're using GitHub, you can trigger the workflow using the GitHub CLI:

```bash
# Authenticate with GitHub
gh auth login

# Trigger the workflow
gh workflow run "🌾 Real USA Agricultural Detection CI/CD Pipeline" --ref main --field environment=development
```

Or you can manually trigger it from the Actions tab in your GitHub repository.

### Local Simulation

For local simulation of the workflow, you can use the provided scripts:

**PowerShell (Windows):**
```powershell
.\run_cicd.ps1
```

**Bash (Linux/macOS):**
```bash
bash ./run_cicd.sh
```

## Pipeline Configuration

The pipeline is defined in `.github/workflows/enhanced-ci-cd-pipeline.yml`. Key configuration options:

- **Environments:** development (default), staging, production
- **Python Version:** 3.11
- **Node Version:** 18
- **Docker Registry:** GitHub Container Registry (ghcr.io)

## Customization

To customize the pipeline for your specific needs:

1. Modify the environment variables at the top of the workflow file
2. Add/remove jobs as needed
3. Adjust the deployment configuration for your infrastructure
4. Update the Docker build and push steps for your registry

## Monitoring

You can monitor pipeline runs in the GitHub Actions tab, where you'll find:
- Detailed logs for each job
- Artifacts from the build process
- Performance reports
- Documentation builds