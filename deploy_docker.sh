#!/bin/bash
# Docker Deployment Script for Real USA Agricultural Detection System

echo "🚀 Starting Docker deployment for Enhanced Agricultural Detection System..."

# Build and start all services
echo "🏗️ Building and starting all Docker services..."
docker-compose up --build -d

# Wait for services to initialize
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

# Verify Redis connection
echo "📊 Testing Redis connection..."
docker exec redis-live-agricultural-results redis-cli ping

# Test API endpoints
echo "🧪 Testing API endpoints..."

# Test main API
echo "Testing Enhanced Adaptive Fusion API..."
curl -f http://localhost:8007/ || echo "❌ API not ready"

# Test WebSocket connection
echo "Testing WebSocket connection..."
docker-compose logs enhanced-adaptive-fusion-api | grep "WebSocket" | tail -5

# Display service URLs
echo "✅ Deployment complete! Services available at:"
echo "🌐 Enhanced Dashboard: http://localhost:8080"
echo "🚀 Adaptive Fusion API: http://localhost:8007"
echo "🔧 Preprocessing Service: http://localhost:8008"
echo "👁️ MaskRCNN Service: http://localhost:8009"
echo "✨ RR RT FER Service: http://localhost:8010"
echo "🧠 Adaptive Fusion Service: http://localhost:8011"
echo "✅ Post-processing Service: http://localhost:8012"
echo "📊 Redis Storage: localhost:6379"

echo "📋 To view logs: docker-compose logs -f [service-name]"
echo "🛑 To stop all services: docker-compose down"
echo "🔄 To restart: docker-compose restart"

# Health check all services
echo "🏥 Performing health checks..."
for service in enhanced-adaptive-fusion-api preprocessing-service maskrcnn-service rr-rt-fer-service adaptive-fusion-service postprocessing-service frontend-dashboard
do
    echo "Checking $service..."
    docker-compose exec -T $service curl -f http://localhost:$(docker-compose port $service | cut -d: -f2)/health 2>/dev/null || echo "$service health check failed"
done

echo "🎉 Docker deployment completed successfully!"