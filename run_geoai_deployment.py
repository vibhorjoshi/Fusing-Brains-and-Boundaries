#!/usr/bin/env python3
"""
Run GeoAI Docker Deployment - Real USA Agricultural Detection System

This script automates the deployment of the GeoAI Agricultural Detection System
using Docker Compose. It includes proper integration with the GeoAI library
and ensures all components work together seamlessly.

Usage:
    python run_geoai_deployment.py [--use-gpu] [--force-rebuild]
"""

import os
import sys
import time
import subprocess
import socket
import json
import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Deploy GeoAI Agricultural Detection System with Docker')
parser.add_argument('--use-gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild of all images')
parser.add_argument('--profile', type=str, default='default', choices=['default', 'production'], help='Docker compose profile')
parser.add_argument('--compose-file', type=str, default='docker-compose.integrated.yml', help='Docker compose file')
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
        return None

def check_docker():
    """Check if Docker is installed and running"""
    try:
        version = run_command(['docker', '--version'])
        if version:
            print(f"Docker version: {version}")
        else:
            print("Docker command found but couldn't get version")
            return False
        
        compose_version = run_command(['docker', 'compose', 'version'])
        if compose_version:
            print(f"Docker Compose version: {compose_version}")
        else:
            print("Docker Compose command found but couldn't get version")
            return False
        
        # Check if Docker daemon is running
        try:
            docker_info = run_command(['docker', 'info'])
            if docker_info:
                print("Docker daemon is running")
                return True
            else:
                print("Failed to get Docker info")
                return False
        except Exception as e:
            # Check if we're on Windows
            if os.name == 'nt':
                # Check Docker Desktop service
                try:
                    service_status = run_command('Get-Service com.docker.service | Select-Object -ExpandProperty Status', shell=True)
                    if service_status and service_status.strip().lower() != 'running':
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

def check_nvidia_docker():
    """Check if NVIDIA Docker runtime is available"""
    if not args.use_gpu:
        return False
        
    try:
        output = run_command(['docker', 'info', '--format', '{{json .Runtimes}}'])
        if output:
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
    required_ports = [6379, 8007, 8081, 8501, 9090]
    in_use_ports = []
    
    for port in required_ports:
        if is_port_in_use(port):
            in_use_ports.append(port)
            
    if in_use_ports:
        print(f"Warning: The following ports are already in use: {in_use_ports}")
        print("This may cause conflicts with the Docker services.")
        response = input("Do you want to continue anyway? (y/n): ").strip().lower()
        return response == 'y'
    else:
        print("All required ports are available")
        return True

def check_geoai_library():
    """Check if the GeoAI library is available"""
    geoai_path = os.path.join('src', 'open_source_geo_ai.py')
    if os.path.exists(geoai_path):
        print(f"✅ GeoAI library found: {geoai_path}")
        return True
    else:
        print(f"❌ GeoAI library not found: {geoai_path}")
        print("The system may not function correctly without it.")
        return False

def build_and_start_containers():
    """Build and start Docker containers"""
    compose_file = args.compose_file
    
    # Ensure the compose file exists
    if not os.path.exists(compose_file):
        print(f"Error: Docker Compose file '{compose_file}' not found")
        return False
    
    print(f"Using Docker Compose file: {compose_file}")
    
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
    if not run_command(build_cmd):
        print("Error building Docker images")
        return False
    
    print("\nStarting containers...")
    if not run_command(start_cmd):
        print("Error starting Docker containers")
        return False
    
    return True

def monitor_container_startup():
    """Monitor container startup and health"""
    print("\nMonitoring container startup...")
    
    # Wait a moment for containers to start
    time.sleep(5)
    
    # Get list of containers
    containers = run_command(['docker', 'compose', '-f', args.compose_file, 'ps', '--format', 'json'])
    
    if not containers:
        print("No containers found")
        return False
    
    # Parse JSON (each line is a JSON object)
    container_list = []
    for line in containers.strip().split('\n'):
        if line:
            try:
                container_list.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Print status
    print("\nContainer Status:")
    print("-" * 80)
    print(f"{'Name':<30} {'Status':<20} {'Health':<20}")
    print("-" * 80)
    
    all_healthy = True
    for container in container_list:
        name = container.get('Name', 'Unknown')
        status = container.get('State', 'Unknown')
        health = container.get('Health', 'N/A')
        print(f"{name:<30} {status:<20} {health:<20}")
        if status.lower() != 'running':
            all_healthy = False
    
    return all_healthy

def print_access_info():
    """Print access information for services"""
    print("\nAccess Information:")
    print("-" * 80)
    print("Frontend Dashboard:    http://localhost:8081")
    print("Streamlit Analytics:   http://localhost:8501")
    print("API Endpoint:          http://localhost:8007")
    print("Monitoring Metrics:    http://localhost:9090/metrics")
    print("-" * 80)
    print("\nTo stop the containers, run:")
    print(f"docker compose -f {args.compose_file} down")
    print("-" * 80)

def open_dashboard_in_browser():
    """Open the dashboards in a web browser"""
    try:
        # Open frontend in default browser
        webbrowser.open(f"http://localhost:8081")
        
        # Wait a bit before opening Streamlit
        time.sleep(2)
        
        # Open Streamlit in new tab
        webbrowser.open(f"http://localhost:8501")
    except Exception as e:
        print(f"Error opening browser: {e}")

def main():
    """Main function"""
    print("=" * 80)
    print("GeoAI Agricultural Detection System - Docker Deployment")
    print("=" * 80)
    
    # Check Docker
    if not check_docker():
        print("Docker check failed. Please fix the issues and try again.")
        return False
    
    # Check NVIDIA Docker
    has_nvidia = check_nvidia_docker()
    
    # Check GeoAI library
    check_geoai_library()
    
    # Check ports
    if not check_ports():
        return False
    
    # Build and start containers
    if not build_and_start_containers():
        return False
    
    # Monitor container startup
    all_healthy = monitor_container_startup()
    if not all_healthy:
        print("\nWarning: Not all containers are running properly.")
        print("Check the container logs for more information:")
        print(f"docker compose -f {args.compose_file} logs")
    
    # Print access information
    print_access_info()
    
    # Open dashboards in browser
    open_dashboard_in_browser()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)