@echo off
REM Docker Deployment Script for Real USA Agricultural Detection System

echo 🚀 Starting Docker deployment for Enhanced Agricultural Detection System...

REM Build and start all services
echo 🏗️ Building and starting all Docker services...
docker-compose up --build -d

REM Wait for services to initialize
echo ⏳ Waiting for services to initialize...
timeout /t 30 /nobreak > nul

REM Check service status
echo 🔍 Checking service status...
docker-compose ps

REM Verify Redis connection
echo 📊 Testing Redis connection...
docker exec redis-live-agricultural-results redis-cli ping

REM Test API endpoints
echo 🧪 Testing API endpoints...
curl -f http://localhost:8007/ || echo ❌ API not ready

REM Display service URLs
echo.
echo ✅ Deployment complete! Services available at:
echo 🌐 Enhanced Dashboard: http://localhost:8080
echo 🚀 Adaptive Fusion API: http://localhost:8007
echo 🔧 Preprocessing Service: http://localhost:8008
echo 👁️ MaskRCNN Service: http://localhost:8009
echo ✨ RR RT FER Service: http://localhost:8010
echo 🧠 Adaptive Fusion Service: http://localhost:8011
echo ✅ Post-processing Service: http://localhost:8012
echo 📊 Redis Storage: localhost:6379
echo.
echo 📋 To view logs: docker-compose logs -f [service-name]
echo 🛑 To stop all services: docker-compose down
echo 🔄 To restart: docker-compose restart
echo.
echo 🎉 Docker deployment completed successfully!
echo.
pause