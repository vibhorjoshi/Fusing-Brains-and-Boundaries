# GeoAI Deployment Guide

This guide provides instructions for deploying the GeoAI Research Project components to various platforms including Docker, Vercel, and cloud providers.

## Table of Contents

- [Local Docker Deployment](#local-docker-deployment)
- [Vercel Frontend Deployment](#vercel-frontend-deployment)
- [Environment Configuration](#environment-configuration)
- [CI/CD Pipeline](#cicd-pipeline)
- [Production Scaling](#production-scaling)

## Local Docker Deployment

The GeoAI system uses Docker containers for easy deployment of all components including the backend API, GeoAI engine, monitoring, and database services.

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available
- 20GB free disk space

### Deployment Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd geo-ai-research-paper
   ```

2. Run the Docker deployment script:
   ```bash
   python run_docker_deployment.py
   ```

3. Access the services:
   - Streamlit Dashboard: http://localhost:8501
   - GeoAI API: http://localhost:8000
   - Monitoring Dashboard: http://localhost:3000

### Customizing Docker Deployment

To customize the deployment, edit the `docker-compose.geoai-enhanced.yml` file to adjust resource limits, ports, or add additional services.

## Vercel Frontend Deployment

The frontend application can be deployed to Vercel for improved global access and performance.

### Prerequisites

- Node.js 18+ installed
- Vercel account
- Vercel CLI installed (`npm install -g vercel`)

### Deployment Steps

1. Configure environment variables:
   - Create `.env.production` in the frontend directory
   - Set backend API URLs and other required variables

2. Run the Vercel deployment script:
   ```bash
   node deploy_to_vercel.js --production
   ```

3. For preview deployments (non-production):
   ```bash
   node deploy_to_vercel.js
   ```

### Manual Deployment

If you prefer to deploy manually:

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Log in to Vercel CLI:
   ```bash
   vercel login
   ```

3. Deploy:
   ```bash
   # For preview deployment
   vercel
   
   # For production deployment
   vercel --prod
   ```

## Environment Configuration

### Frontend (.env.production)

```
NEXT_PUBLIC_API_URL=https://your-api-url.com
NEXT_PUBLIC_GEOAI_VERSION=1.0.0
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_google_maps_key
```

### Backend (Docker .env)

```
DATABASE_URL=postgresql://user:password@db:5432/geoai
SECRET_KEY=your_secret_key
DEBUG=False
ALLOWED_HOSTS=*
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically builds, tests, and deploys the application.

### Workflow Steps

1. **Test**: Runs unit and integration tests
2. **Build**: Builds Docker images and frontend assets
3. **Deploy**: Deploys to staging or production based on branch

### Configuration

The CI/CD pipeline is configured in `.github/workflows/geoai-ci-cd.yml`. You can customize the workflow by:

- Modifying test requirements
- Adding additional deployment targets
- Configuring environment-specific variables

## Production Scaling

### Docker Swarm/Kubernetes

For production environments, consider using Docker Swarm or Kubernetes for:

- Horizontal scaling of services
- Load balancing
- Zero-downtime deployments
- Health monitoring and auto-recovery

### Database Scaling

- Use connection pooling
- Consider read replicas for heavy read operations
- Implement proper indexing strategies

### Monitoring

Monitor system performance with the built-in Prometheus/Grafana stack:

- Track API response times
- Monitor resource usage
- Set up alerts for critical thresholds

---

For additional support or questions, please refer to the documentation or open an issue on GitHub.