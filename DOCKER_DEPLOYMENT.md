# GeoAI Agricultural Detection System - Docker Deployment

This guide explains how to deploy the Real USA Agricultural Detection System using Docker containers.

## Components

The system consists of the following components:

1. **Backend API** - Enhanced Adaptive Fusion API server (FastAPI)
2. **MaskRCNN Service** - 3D mask generation for agricultural detection
3. **Streamlit Dashboard** - Interactive data visualization
4. **Redis** - Storage and caching layer
5. **Performance Monitor** - System metrics and container stats

## Prerequisites

- Docker and Docker Compose installed
- For GPU support: NVIDIA Docker runtime
- Recommended: At least 8GB RAM and 4 CPU cores
- Storage: At least 10GB free space

## Quick Start

The easiest way to deploy the system is using the provided script:

```bash
# Start the deployment
python run_docker_deployment.py

# For GPU support:
python run_docker_deployment.py --use-gpu

# To force rebuild all images:
python run_docker_deployment.py --force-rebuild

# For production deployment (includes Nginx):
python run_docker_deployment.py --profile production
```

## Manual Deployment

If you prefer to run the Docker commands manually:

```bash
# Build the images
docker compose -f docker-compose.updated.yml build

# Start the containers
docker compose -f docker-compose.updated.yml up -d

# View logs
docker compose -f docker-compose.updated.yml logs -f

# Stop the containers
docker compose -f docker-compose.updated.yml down
```

## Access Points

Once deployed, the services will be available at:

- Frontend UI: http://localhost:3000
- Streamlit Dashboard: http://localhost:8501
- API Endpoint: http://localhost:8002
- Monitoring: http://localhost:9090/metrics

## AWS Deployment

To deploy the system to AWS:

```bash
# Deploy to AWS ECS (Elastic Container Service)
python deploy_to_aws.py --region us-east-1

# Deploy to AWS EC2
python deploy_to_aws.py --region us-east-1 --deploy-mode ec2
```

## Environment Variables

The following environment variables can be customized:

- `REDIS_HOST`: Redis server hostname (default: redis)
- `REDIS_PORT`: Redis server port (default: 6379)
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)

## Troubleshooting

If you encounter issues with the deployment:

1. Check container status: `docker ps -a`
2. View container logs: `docker logs <container-name>`
3. Check Redis connection: `docker exec geoai-redis redis-cli ping`
4. Ensure required ports are available
5. For GPU issues, check NVIDIA runtime: `docker info | grep -i nvidia`

## License

This project is licensed under the MIT License - see the LICENSE file for details.