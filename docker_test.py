#!/usr/bin/env python3
"""
Docker Environment Test Script
This script helps diagnose Docker and Docker Compose issues
"""

import os
import sys
import subprocess
import platform
import json

def run_test_command(command, description):
    """Run a test command and return results"""
    print(f"\n=== {description} ===")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            text=True,
            capture_output=True
        )
        
        success = result.returncode == 0
        print(f"Exit code: {result.returncode} ({'SUCCESS' if success else 'FAILED'})")
        
        if result.stdout:
            print("\nOutput:")
            print(result.stdout.strip())
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr.strip())
            
        return {
            "success": success,
            "exit_code": result.returncode,
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        print(f"Error executing command: {str(e)}")
        return {
            "success": False,
            "exit_code": -1,
            "output": "",
            "error": str(e)
        }

def main():
    """Run Docker environment tests"""
    print("=" * 80)
    print(" DOCKER ENVIRONMENT DIAGNOSTIC TOOL")
    print("=" * 80)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print("-" * 80)
    
    # Test Docker installation
    docker_version = run_test_command("docker --version", "Docker Version")
    docker_info = run_test_command("docker info", "Docker System Info")
    
    # Test Docker Compose installation
    compose_version = run_test_command("docker-compose --version", "Docker Compose Version")
    
    # Test basic Docker functionality
    hello_world = run_test_command("docker run --rm hello-world", "Hello World Container Test")
    
    # Test Docker Compose configuration
    compose_files = []
    for file_name in ["docker-compose.yml", "docker-compose.dev.yml", "docker-compose.minimal.yml"]:
        if os.path.exists(file_name):
            compose_files.append(file_name)
            
    for compose_file in compose_files:
        run_test_command(f"docker-compose -f {compose_file} config", f"Validating {compose_file}")
    
    # If no compose file found, create a minimal test file
    if not compose_files:
        print("\n=== Creating Minimal Test Compose File ===")
        minimal_content = """version: '3'
services:
  test:
    image: hello-world
"""
        try:
            with open("docker-compose.test.yml", "w") as f:
                f.write(minimal_content)
            print("Created docker-compose.test.yml")
            run_test_command("docker-compose -f docker-compose.test.yml config", "Validating test compose file")
        except Exception as e:
            print(f"Error creating test file: {str(e)}")
    
    # Check Docker network
    run_test_command("docker network ls", "Docker Network List")
    
    # Check Docker volumes
    run_test_command("docker volume ls", "Docker Volume List")
    
    # Summary
    print("\n" + "=" * 80)
    print(" DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    print(f"Docker installed: {'YES' if docker_version['success'] else 'NO'}")
    print(f"Docker running: {'YES' if docker_info['success'] else 'NO'}")
    print(f"Docker Compose installed: {'YES' if compose_version['success'] else 'NO'}")
    print(f"Basic Docker functionality: {'WORKING' if hello_world['success'] else 'FAILED'}")
    
    if not docker_version['success']:
        print("\n⚠️ Docker is not installed or not in PATH")
        print("   Download from: https://www.docker.com/products/docker-desktop")
    
    if docker_version['success'] and not docker_info['success']:
        print("\n⚠️ Docker is installed but not running")
        print("   Start Docker Desktop or the Docker service")
    
    if not compose_version['success']:
        print("\n⚠️ Docker Compose is not installed or not in PATH")
        print("   It should be included with Docker Desktop")
    
    if not hello_world['success']:
        print("\n⚠️ Basic Docker functionality test failed")
        print("   Check Docker installation and permissions")
    
    # Deployment recommendation
    print("\n" + "-" * 80)
    if docker_version['success'] and docker_info['success'] and compose_version['success'] and hello_world['success']:
        print("✅ Docker environment appears to be working correctly")
        print("   Recommended command: python deploy.py --env development")
    else:
        print("❌ Docker environment has issues")
        print("   Recommended command: python deploy.py --env development --local")
    print("-" * 80)
    
if __name__ == "__main__":
    main()