#!/usr/bin/env python3
"""
Run Docker Deployment - Real USA Agricultural Detection System

This script automates the deployment of the GeoAI Agricultural Detection System
using Docker Compose with independent microservices. It will:
1. Check for Docker installation and requirements
2. Build and start the containerized microservices defined in docker-compose.yml
3. Monitor container startup, health, and live performance metrics
4. Print access URLs for all services including the GeoAI dashboard
5. Configure continuous monitoring for 24/7 operation

Usage:
    python run_docker_deployment.py [--use-gpu] [--force-rebuild] [--profile PROFILE]
"""

import os
import sys
import time
import subprocess
import socket
import json
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Deploy GeoAI Agricultural Detection System with Docker')
parser.add_argument('--use-gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild of all images')
parser.add_argument('--profile', type=str, default='default', choices=['default', 'production', 'development'], help='Docker compose profile')
parser.add_argument('--compose-file', type=str, default='docker-compose.geoai-enhanced.yml', help='Docker compose file')
parser.add_argument('--monitoring-only', action='store_true', help='Start only monitoring containers')
parser.add_argument('--update-from-github', action='store_true', help='Update containers from GitHub container registry')
args = parser.parse_args()

def run_command(command, shell=False):
    """Run a shell command and return output"""
    try:
        if shell:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        else:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def check_docker():
    """Check if Docker is installed and running"""
    try:
        version = run_command(['docker', '--version'])
        print(f"Docker version: {version}")
        
        compose_version = run_command(['docker', 'compose', 'version'])
        print(f"Docker Compose version: {compose_version}")
        
        # Check if Docker daemon is running
        try:
            run_command(['docker', 'info'])
            return True
        except Exception as e:
            # Check if we're on Windows
            if os.name == 'nt':
                # Check Docker Desktop service
                try:
                    service_status = run_command('Get-Service com.docker.service | Select-Object -ExpandProperty Status', shell=True)
                    if service_status.strip().lower() != 'running':
                        print("\nERROR: Docker Desktop is not running!")
                        print("Please start Docker Desktop application and wait for it to initialize.")
                        print("Windows solution steps:")
                        print("1. Search for 'Docker Desktop' in the Start menu and launch it")
                        print("2. Wait for Docker Desktop to fully start (check the system tray icon)")
                        print("3. Run this script again\n")
                        return False
                except Exception:
                    pass
            
            print(f"\nERROR: Docker daemon is not running: {e}")
            print("Make sure Docker is installed and running correctly before continuing.\n")
            return False
            
    except Exception as e:
        print(f"\nERROR: Docker not found: {e}")
        print("Please install Docker Desktop from https://www.docker.com/products/docker-desktop\n")
        return False
        
def check_required_files():
    """Check if all required files for deployment exist"""
    required_files = [
        args.compose_file,
        "monitoring/prometheus.yml",
        "monitoring/grafana/provisioning/dashboards/dashboards.yml",
        "monitoring/grafana/provisioning/datasources/datasource.yml",
        "nginx/conf/nginx.conf",
        "nginx/conf/geoai.conf"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("\nERROR: The following required files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure all required files are present before continuing.\n")
        return False
        
    return True

def check_nvidia_docker():
    """Check if NVIDIA Docker runtime is available"""
    if not args.use_gpu:
        return False
        
    try:
        output = run_command(['docker', 'info', '--format', '{{json .Runtimes}}'])
        runtimes = json.loads(output)
        if 'nvidia' in runtimes:
            print("NVIDIA Docker runtime is available")
            return True
    except Exception:
        pass
        
    print("NVIDIA Docker runtime is not available, using CPU")
    return False

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_ports():
    """Check if required ports are available"""
    required_ports = [6379, 8000, 8002, 8501, 3000, 9090]
    in_use_ports = []
    
    for port in required_ports:
        if is_port_in_use(port):
            in_use_ports.append(port)
            
    if in_use_ports:
        print(f"Warning: The following ports are already in use: {in_use_ports}")
        print("This may cause conflicts with the Docker services.")
        input("Press Enter to continue or Ctrl+C to abort...")
    else:
        print("All required ports are available")

def build_and_start_containers():
    """Build and start Docker containers"""
    compose_file = args.compose_file
    
    # Ensure the compose file exists
    if not os.path.exists(compose_file):
        print(f"Error: Docker Compose file '{compose_file}' not found")
        sys.exit(1)
    
    print(f"Using Docker Compose file: {compose_file}")
    
    # Create necessary directories
    required_dirs = [
        "monitoring/grafana/dashboards",
        "monitoring/grafana/provisioning/dashboards",
        "monitoring/grafana/provisioning/datasources",
        "nginx/conf",
        "nginx/certs",
        "nginx/logs"
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Check if using enhanced configuration
    if compose_file == 'docker-compose.geoai-enhanced.yml':
        print("Using enhanced GeoAI configuration with independent microservices")
        print("Setting up 24/7 monitoring and live performance dashboards")
    
    # Build command
    build_cmd = ['docker', 'compose', '-f', compose_file]
    if args.profile != 'default':
        build_cmd.extend(['--profile', args.profile])
    if args.force_rebuild:
        build_cmd.extend(['build', '--no-cache'])
    else:
        build_cmd.append('build')
    
    # Start command
    start_cmd = ['docker', 'compose', '-f', compose_file]
    if args.profile != 'default':
        start_cmd.extend(['--profile', args.profile])
    start_cmd.extend(['up', '-d'])
    
    # Run commands
    print("Building Docker images (this may take a while)...")
    run_command(build_cmd)
    
    print("\nStarting containers...")
    run_command(start_cmd)

def monitor_container_startup():
    """Monitor container startup and health"""
    print("\nMonitoring container startup...")
    
    # Wait a moment for containers to start
    time.sleep(5)
    
    # Get list of containers
    containers = run_command(['docker', 'compose', '-f', args.compose_file, 'ps', '--format', 'json'])
    
    if not containers:
        print("No containers found")
        return
    
    # Parse JSON (each line is a JSON object)
    container_list = []
    for line in containers.strip().split('\n'):
        if line:
            container_list.append(json.loads(line))
    
    # Print status
    print("\nContainer Status:")
    print("-" * 80)
    print(f"{'Name':<30} {'Status':<20} {'Health':<20}")
    print("-" * 80)
    
    for container in container_list:
        name = container.get('Name', 'Unknown')
        status = container.get('State', 'Unknown')
        health = container.get('Health', 'N/A')
        print(f"{name:<30} {status:<20} {health:<20}")

def print_access_info():
    """Print access information for the services"""
    host = "localhost"
    
    print("\nAccess Information:")
    print("=" * 80)
    print(f"Main Application: http://{host}")
    print(f"API Documentation: http://{host}/api/docs")
    print(f"GeoAI Engine: http://{host}/geoai/docs")
    print(f"Streamlit Dashboard: http://{host}/dashboard")
    print(f"Performance Monitoring: http://{host}/monitoring")
    print("=" * 80)
    
    print("\nContainer Services:")
    print("=" * 80)
    print(f"Frontend: http://{host}:3000")
    print(f"Backend API: http://{host}:8002")
    print(f"GeoAI Engine: http://{host}:8003")
    print(f"Streamlit Dashboard: http://{host}:8501")
    print(f"Grafana Dashboard: http://{host}:3001")
    print(f"Prometheus Metrics: http://{host}:9090")
    print("=" * 80)
    
    print("\n24/7 Monitoring Status:")
    print("=" * 80)
    try:
        result = run_command(['docker', 'compose', '-f', args.compose_file, 'ps', '--format', 'json'])
        
        # Parse JSON (each line is a JSON object)
        container_list = []
        for line in result.strip().split('\n'):
            if line:
                container_list.append(json.loads(line))
                
        for container in container_list:
            name = container.get('Name', 'Unknown')
            status = container.get('State', 'Unknown')
            health = container.get('Health', 'N/A')
            
            status_text = f"{status}"
            if health and health != "N/A":
                status_text += f" ({health})"
                
            print(f"{name}: {status_text}")
            
    except Exception as e:
        print(f"Could not retrieve container status: {e}")
        
    print("=" * 80)
    
    print("\nTo stop the containers, run:")
    print(f"docker compose -f {args.compose_file} down")
    print("=" * 80)

def main():
    """Main function"""
    print("=" * 80)
    print("GeoAI Agricultural Detection System - Docker Deployment")
    print("=" * 80)
    
    # Check Docker
    if not check_docker():
        sys.exit(1)
    
    # Check for required files
    if not check_required_files():
        sys.exit(1)
    
    # Check NVIDIA Docker
    has_nvidia = check_nvidia_docker()
    
    # Set environment variable for GPU usage
    if has_nvidia:
        os.environ["USE_GPU"] = "true"
    else:
        os.environ["USE_GPU"] = "false"
    
    # Check ports
    check_ports()
    
    # Build and start containers
    build_and_start_containers()
    
    # Monitor container startup
    monitor_container_startup()
    
    # Print access information
    print_access_info()
    
    print("\nGeoAI Live Performance Dashboard is now running 24/7!")
    print("This system provides detailed USA map visualizations with real-time updates.")
    print("Access the Grafana dashboard for live performance metrics and system health.")
    print("=" * 80)

if __name__ == "__main__":
    main()