#!/bin/bash
# GeoAI Platform Complete Startup Script

set -e

echo "🚀 Starting GeoAI NASA-Level Platform..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if Python virtual environment exists
check_python_env() {
    print_header "Checking Python Environment"
    
    if [ ! -d ".venv" ]; then
        print_warning "Python virtual environment not found. Creating..."
        python -m venv .venv
    fi
    
    print_status "Activating Python virtual environment"
    source .venv/bin/activate || source .venv/Scripts/activate
    
    print_status "Python environment activated"
    python --version
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"
    
    print_status "Upgrading pip"
    python -m pip install --upgrade pip
    
    print_status "Installing backend dependencies"
    pip install -r requirements.txt
    
    print_status "Python dependencies installed successfully"
}

# Install Node.js dependencies
install_node_deps() {
    print_header "Installing Frontend Dependencies"
    
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        print_status "Installing Node.js dependencies"
        npm install
    else
        print_status "Node.js dependencies already installed"
    fi
    
    cd ..
    print_status "Frontend dependencies ready"
}

# Start backend server
start_backend() {
    print_header "Starting Backend Server"
    
    print_status "Starting FastAPI server on port 8002"
    python simple_demo_server.py &
    BACKEND_PID=$!
    
    # Wait for backend to start
    sleep 5
    
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_status "Backend server started successfully (PID: $BACKEND_PID)"
        echo $BACKEND_PID > .backend.pid
    else
        print_error "Failed to start backend server"
        exit 1
    fi
}

# Start frontend server
start_frontend() {
    print_header "Starting Frontend Server"
    
    cd frontend
    
    print_status "Starting Next.js development server on port 3000"
    npm run dev &
    FRONTEND_PID=$!
    
    cd ..
    
    # Wait for frontend to start
    sleep 10
    
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        print_status "Frontend server started successfully (PID: $FRONTEND_PID)"
        echo $FRONTEND_PID > .frontend.pid
    else
        print_error "Failed to start frontend server"
        exit 1
    fi
}

# Health check
health_check() {
    print_header "Performing Health Checks"
    
    # Check backend
    print_status "Checking backend health..."
    if curl -f http://127.0.0.1:8002/health >/dev/null 2>&1; then
        print_status "✅ Backend is healthy"
    else
        print_warning "⚠️  Backend health check failed"
    fi
    
    # Check frontend
    print_status "Checking frontend health..."
    if curl -f http://localhost:3000 >/dev/null 2>&1; then
        print_status "✅ Frontend is healthy"
    else
        print_warning "⚠️  Frontend health check failed"
    fi
}

# Display access URLs
show_urls() {
    print_header "🎉 GeoAI Platform is Running!"
    
    echo ""
    echo "📍 Access URLs:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 Frontend (NASA Interface): http://localhost:3000"
    echo "🚀 Backend API:              http://127.0.0.1:8002"
    echo "📖 API Documentation:        http://127.0.0.1:8002/docs"
    echo "🌟 Live Visualization:       http://127.0.0.1:8002/live"
    echo "🌍 Globe View:               http://127.0.0.1:8002/globe"
    echo "🧠 ML Pipeline:              http://127.0.0.1:8002/ml-processing"
    echo "📊 Analytics:                http://127.0.0.1:8002/analytics"
    echo ""
    echo "🔧 System Status:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Backend PID:  $(cat .backend.pid 2>/dev/null || echo 'Not available')"
    echo "Frontend PID: $(cat .frontend.pid 2>/dev/null || echo 'Not available')"
    echo ""
    echo "💡 To stop the platform, run: ./stop_platform.sh"
    echo "🔍 To check logs, run: tail -f logs/*.log"
}

# Cleanup function
cleanup() {
    print_header "Shutting down GeoAI Platform"
    
    if [ -f .backend.pid ]; then
        BACKEND_PID=$(cat .backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            print_status "Stopping backend server (PID: $BACKEND_PID)"
            kill $BACKEND_PID
        fi
        rm -f .backend.pid
    fi
    
    if [ -f .frontend.pid ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            print_status "Stopping frontend server (PID: $FRONTEND_PID)"
            kill $FRONTEND_PID
        fi
        rm -f .frontend.pid
    fi
    
    print_status "Platform shutdown complete"
    exit 0
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    # Create logs directory
    mkdir -p logs
    
    # Check dependencies
    command -v python3 >/dev/null 2>&1 || { print_error "Python 3 is required but not installed."; exit 1; }
    command -v npm >/dev/null 2>&1 || { print_error "Node.js/npm is required but not installed."; exit 1; }
    command -v curl >/dev/null 2>&1 || { print_error "curl is required but not installed."; exit 1; }
    
    # Setup and start services
    check_python_env
    install_python_deps
    install_node_deps
    start_backend
    start_frontend
    
    # Wait for services to fully initialize
    sleep 15
    
    # Perform health checks
    health_check
    
    # Show access information
    show_urls
    
    # Keep script running
    print_status "Platform is running. Press Ctrl+C to stop."
    while true; do
        sleep 30
        # Optional: Periodic health checks
        if ! kill -0 $(cat .backend.pid 2>/dev/null) 2>/dev/null || ! kill -0 $(cat .frontend.pid 2>/dev/null) 2>/dev/null; then
            print_error "One or more services have stopped unexpectedly"
            cleanup
        fi
    done
}

# Run main function
main "$@"