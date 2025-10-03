#!/usr/bin/env python3
"""
Deployment Script for Real USA Agricultural Detection System
This script helps deploy the system to various environments
"""

import os
import sys
import argparse
import subprocess
import platform
import time
import json
from datetime import datetime

# Try to import dotenv, install if not available
try:
    from dotenv import load_dotenv
except ImportError:
    print("Installing python-dotenv package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], check=False)
    from dotenv import load_dotenv

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Back, Style
    init()  # Initialize colorama
    has_colorama = True
except ImportError:
    has_colorama = False

def colored(text, color):
    """Apply color to text if colorama is available"""
    if has_colorama:
        color_map = {
            "red": Fore.RED,
            "green": Fore.GREEN,
            "yellow": Fore.YELLOW,
            "blue": Fore.BLUE,
            "magenta": Fore.MAGENTA,
            "cyan": Fore.CYAN,
            "white": Fore.WHITE
        }
        return f"{color_map.get(color, '')}{text}{Style.RESET_ALL}"
    else:
        # Fallback if colorama is not available
        return text

# Define environment options
ENVIRONMENTS = ["development", "staging", "production"]

# Global settings
VERBOSE = False

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")

def run_command(command, description=None, check=True, verbose=None):
    """Run a shell command and print output with enhanced diagnostics"""
    global VERBOSE
    
    # Use provided verbose flag or global setting
    use_verbose = VERBOSE if verbose is None else verbose
    
    if description:
        print(f"🚀 {description}...")
    
    if use_verbose:
        print(f"Command: {command}")
    
    try:
        # For Windows, ensure the shell is powershell for better compatibility
        shell_to_use = True  # Default to system shell
        if platform.system() == "Windows":
            # Check if we're in PowerShell
            if os.environ.get('PSModulePath'):
                # We're in PowerShell
                shell_to_use = True
            else:
                # Force PowerShell if not in it already
                command = f'powershell -Command "{command}"'
        
        # Run the command
        start_time = time.time()
        result = subprocess.run(
            command, 
            shell=shell_to_use, 
            check=False,  # Don't raise exception here, handle it below
            text=True,
            capture_output=True
        )
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Format output more clearly
        if use_verbose:
            print(f"Command completed in {execution_time:.2f} seconds with exit code: {result.returncode}")
        
        # Handle standard output
        if result.stdout:
            if use_verbose:
                print("\n--- Standard Output ---")
                print(result.stdout.strip())
                print("--- End Output ---\n")
            elif result.returncode != 0:
                # Show output on error even if not in verbose mode
                print("Output:")
                print(result.stdout.strip())
        elif use_verbose:
            print("(No standard output)")
        
        # Handle standard error
        if result.stderr:
            if use_verbose or result.returncode != 0:
                print("\n--- Error Output ---")
                print(result.stderr.strip())
                print("--- End Error Output ---\n")
        elif use_verbose and result.returncode != 0:
            print("(No error output despite non-zero return code)")
        
        # Check return code
        if result.returncode != 0:
            if check:
                print(f"❌ Command failed with exit code {result.returncode}")
                
                # Add more diagnostics for common exit codes
                if result.returncode == 127:
                    print("This typically means the command was not found.")
                elif result.returncode == 126:
                    print("This typically means the command was found but couldn't be executed.")
                elif result.returncode == 1:
                    print("This is a general error condition, check the output for details.")
                
                return False
            elif use_verbose:
                print(f"⚠️ Command exited with non-zero code {result.returncode} (ignored due to check=False)")
        else:
            if use_verbose:
                print("✅ Command completed successfully")
        
        return result.returncode == 0
    except subprocess.SubprocessError as e:
        print(f"❌ Subprocess error: {str(e)}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: The system cannot find the specified command.")
        return False
    except Exception as e:
        print(f"❌ Command execution error: {str(e)}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("Checking Dependencies")
    
    dependencies = {
        "docker": "Docker must be installed. Download from https://www.docker.com/",
        "docker-compose": "Docker Compose must be installed. Usually comes with Docker."
    }
    
    all_good = True
    
    # Check for Docker executable
    if platform.system() == "Windows":
        # On Windows, use Get-Command
        docker_command = "powershell -Command \"if (Get-Command docker -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }\""
    else:
        docker_command = "docker --version"
    
    if run_command(docker_command, "Checking for Docker", check=False):
        print("✅ Docker executable is found")
        
        # Check if Docker service is running
        if run_command("docker info", "Checking if Docker service is running", check=False):
            print("✅ Docker service is running")
            
            # Get Docker version details
            run_command("docker version", "Docker version details", check=False)
        else:
            print("❌ Docker service is not running")
            print("   Please start Docker Desktop or the Docker service")
            all_good = False
    else:
        print("❌ Docker executable not found - Docker must be installed.")
        print("   Download from https://www.docker.com/")
        all_good = False
    
    # Check for Docker Compose
    if platform.system() == "Windows":
        # On Windows, use Get-Command
        compose_command = "powershell -Command \"if (Get-Command docker-compose -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }\""
    else:
        compose_command = "docker-compose --version"
    
    if run_command(compose_command, "Checking for Docker Compose", check=False):
        print("✅ Docker Compose is installed")
        
        # Check Docker Compose version
        run_command("docker-compose version", "Docker Compose version details", check=False)
    else:
        print("❌ Docker Compose not found")
        print("   It should be included with Docker Desktop, or can be installed separately")
        all_good = False
    
    # If Docker is running, check docker network to ensure networking is working
    if all_good:
        print("\nVerifying Docker networking:")
        run_command("docker network ls", "Listing Docker networks", check=False)
    
    return all_good

def check_environment(env):
    """Check environment-specific requirements"""
    print_header(f"Checking {env.upper()} Environment Configuration")
    
    # Check for environment-specific configuration
    env_file = f".env.{env}"
    if not os.path.exists(env_file):
        print(f"⚠️ Environment file {env_file} not found, creating default")
        with open(env_file, "w") as f:
            f.write(f"ENVIRONMENT={env}\n")
            f.write("REDIS_URL=redis://redis:6379\n")
            f.write("API_PORT=8002\n")
            f.write("USE_GPU=false\n")
    else:
        print(f"✅ Found environment file: {env_file}")
    
    return True

def stop_running_containers():
    """Stop any running containers that might conflict"""
    print_header("Stopping Running Containers")
    
    run_command("docker-compose down -v", "Stopping containers")
    return True

def pull_latest_images(env):
    """Pull the latest Docker images"""
    print_header("Pulling Latest Images")
    
    if env == "production":
        run_command("docker-compose pull", "Pulling production images")
    else:
        print("⏩ Skipping image pull for non-production environment")
    
    return True

def start_services(env):
    """Start the services with Docker Compose"""
    print_header("Starting Services")
    
    # Check if docker-compose is available
    if not run_command("docker-compose --version", "Checking Docker Compose installation", check=False):
        print("❌ Docker Compose not found. Please install Docker Compose first.")
        return False
        
    # Check if Docker is running
    if not run_command("docker info", "Checking Docker service status", check=False):
        print("❌ Docker service is not running. Please start Docker first.")
        return False
    
    # Use environment-specific compose file if available
    compose_files = []
    
    # Check for environment-specific file
    if os.path.exists(f"docker-compose.{env}.yml"):
        compose_files.append(f"docker-compose.{env}.yml")
        print(f"✅ Found environment-specific compose file: docker-compose.{env}.yml")
    
    # Check for default compose file
    if os.path.exists("docker-compose.yml"):
        compose_files.append("docker-compose.yml")
        print("✅ Found default compose file: docker-compose.yml")
    
    # Check for minimal compose file
    if os.path.exists("docker-compose.minimal.yml"):
        compose_files.append("docker-compose.minimal.yml")
        print("✅ Found minimal compose file: docker-compose.minimal.yml")
    
    # If no files found, create minimal
    if not compose_files:
        print("⚠️ No docker-compose files found, creating a minimal one...")
        create_minimal_compose_file()
        compose_files.append("docker-compose.minimal.yml")
    
    # Start with most specific file
    compose_file = compose_files[0]
    print(f"✅ Using compose file: {compose_file}")
    
    # First, validate the Docker Compose file
    if not run_command(f"docker-compose -f {compose_file} config", "Validating docker-compose file", check=False, verbose=True):
        print(f"❌ Invalid docker-compose file: {compose_file}")
        
        # If we have multiple files, try the next one
        if len(compose_files) > 1:
            print(f"⚠️ Trying alternative compose file: {compose_files[1]}")
            compose_file = compose_files[1]
            
            # Validate the alternative file
            if not run_command(f"docker-compose -f {compose_file} config", "Validating alternative compose file", check=False, verbose=True):
                print(f"❌ Invalid alternative compose file: {compose_file}")
                return False
        else:
            return False
    
    # Create environment file if it doesn't exist
    env_file = f".env.{env}"
    env_command = ""
    if os.path.exists(env_file):
        env_command = f"--env-file {env_file}"
        print(f"✅ Using environment file: {env_file}")
    else:
        # Create basic env file with minimum settings
        print(f"⚠️ No {env_file} found, creating a basic one...")
        with open(env_file, "w") as f:
            f.write(f"ENVIRONMENT={env}\n")
            f.write("REDIS_URL=redis://redis:6379\n")
            f.write("API_PORT=8002\n")
            f.write("API_HOST=0.0.0.0\n")
        env_command = f"--env-file {env_file}"
    
    # First, do a basic test with Docker to ensure it's working correctly
    print("🔍 Running a basic Docker test...")
    test_success = run_command("docker run --rm hello-world", "Testing Docker with hello-world", check=False, verbose=True)
    
    if not test_success:
        print("❌ Basic Docker test failed. Docker might not be working correctly.")
        print("Please check Docker installation and permissions.")
        return False
    
    # Try to run docker-compose
    command = f"docker-compose -f {compose_file} {env_command} up -d"
    
    try:
        print(f"🚀 Starting services using: {command}")
        success = run_command(command, f"Starting {env} services", verbose=True)
        
        if not success:
            # If it fails, try to get more detailed error information
            print("⚠️ Failed to start services. Gathering diagnostic information...")
            
            # Check Docker system info
            run_command("docker system info", "Docker system information", check=False, verbose=True)
            
            # List running containers
            run_command("docker ps", "Running containers", check=False, verbose=True)
            
            # Check Docker Compose configuration
            run_command(f"docker-compose -f {compose_file} {env_command} config", "Docker Compose configuration", check=False, verbose=True)
            
            # Check available services
            run_command(f"docker-compose -f {compose_file} {env_command} config --services", "Available services", check=False, verbose=True)
            
            # Check current container status
            run_command(f"docker-compose -f {compose_file} {env_command} ps", "Current container status", check=False, verbose=True)
            
            # Try running without detached mode for more output
            print("⚠️ Running docker-compose with full output for diagnosis:")
            run_command(f"docker-compose -f {compose_file} {env_command} up --no-start", "Validating services", check=False, verbose=True)
            
            # Suggest alternatives
            print("\n🔧 Troubleshooting suggestions:")
            print("1. Check if Docker Desktop is running")
            print("2. Restart Docker service")
            print("3. Try running with the --local flag to run without Docker")
            print("4. Check for port conflicts with existing services")
            
            return False
        
        print("✅ Services started successfully with Docker Compose")
        return True
    except Exception as e:
        print(f"❌ Failed to start services: {e}")
        return False

def verify_services():
    """Verify that services are running correctly"""
    print_header("Verifying Services")
    
    # Wait for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Check docker-compose ps
    run_command("docker-compose ps", "Service status")
    
    # Try to access the API
    success = run_command(
        "curl -s -o /dev/null -w '%{http_code}' http://localhost:8002/health | grep -q 200",
        "Checking API health endpoint",
        check=False
    )
    
    if success:
        print("✅ API is healthy")
    else:
        print("⚠️ API health check failed")
    
    return True

def create_minimal_compose_file():
    """Create a minimal docker-compose file if none exists"""
    if os.path.exists("docker-compose.yml") or os.path.exists("docker-compose.minimal.yml"):
        return False
    
    print("⚠️ No docker-compose file found, creating a minimal version...")
    
    minimal_compose = """version: '3.8'

services:
  # Redis Database
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - geoai-network

  # FastAPI Backend ML Server
  backend:
    image: python:3.11-slim
    working_dir: /app
    command: bash -c "pip install -r requirements.txt && python -m uvicorn enhanced_adaptive_fusion_api:app --host 0.0.0.0 --port 8002 --reload"
    volumes:
      - ./:/app
    ports:
      - "8002:8002"
    environment:
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=development
      - API_HOST=0.0.0.0
      - API_PORT=8002
    networks:
      - geoai-network
    depends_on:
      - redis

  # Streamlit App
  streamlit:
    image: python:3.11-slim
    working_dir: /app
    command: bash -c "pip install -r streamlit_requirements.txt && streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
    volumes:
      - ./:/app
    ports:
      - "8501:8501"
    networks:
      - geoai-network
    depends_on:
      - backend

volumes:
  redis_data:
    driver: local

networks:
  geoai-network:
    driver: bridge
"""
    
    # Write the minimal compose file
    with open("docker-compose.minimal.yml", "w") as f:
        f.write(minimal_compose)
    
    print("✅ Created docker-compose.minimal.yml file")
    return True

def generate_test_data():
    """Generate test data for demo purposes"""
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Create a test file if it doesn't exist
    test_file = "outputs/test_data.json"
    if not os.path.exists(test_file):
        print("⚠️ No test data found, generating minimal test data...")
        test_data = {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": {
                "buildings_detected": 1543,
                "accuracy": 0.92,
                "processing_time_ms": 1850
            }
        }
        
        with open(test_file, "w") as f:
            json.dump(test_data, f, indent=2)
        
        print(f"✅ Created test data file: {test_file}")

def run_services_locally(env):
    """Run services locally without Docker"""
    print_header(f"Running Services Locally ({env})")
    
    # Check Python environment
    print("Checking Python environment...")
    python_cmd = sys.executable
    
    if not python_cmd:
        print("❌ Python executable not found.")
        return False
    
    print(f"✅ Using Python: {python_cmd}")
    
    # Check for required Python packages
    required_packages = ["streamlit", "fastapi", "uvicorn"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing required packages...")
        
        install_cmd = f"{python_cmd} -m pip install {' '.join(missing_packages)}"
        success = run_command(install_cmd, "Installing required packages")
        
        if not success:
            print("❌ Failed to install required packages.")
            print(f"Try manually installing: {install_cmd}")
            return False
        else:
            print("✅ Required packages installed successfully.")
    
    # Install all project dependencies
    print("\n📦 Installing project dependencies...")
    success = run_command(f"{python_cmd} -m pip install -r requirements.txt", "Installing project dependencies")
    
    if not success:
        print("⚠️ Some dependencies might not have installed correctly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create a .env file for the environment if it doesn't exist
    env_file = f".env.{env}"
    if not os.path.exists(env_file):
        print(f"Creating environment file: {env_file}")
        with open(env_file, "w") as f:
            f.write(f"ENVIRONMENT={env}\n")
    
    # Set environment variables for the API server
    os.environ["ENVIRONMENT"] = env
    
    # Get the current directory for proper paths
    current_dir = os.path.abspath(os.getcwd())
    
    # Start API server in the background
    print("\n🚀 Starting API server...")
    if platform.system() == "Windows":
        # For Windows, use proper quoting for paths with spaces
        api_command = f"start \"\" \"{python_cmd}\" -m uvicorn enhanced_adaptive_fusion_api:app --host 0.0.0.0 --port 8002 --reload"
    else:
        # For Unix systems, use a different approach to run in background
        api_command = f"\"{python_cmd}\" -m uvicorn enhanced_adaptive_fusion_api:app --host 0.0.0.0 --port 8002 --reload &"
    
    success = run_command(api_command, "Starting API server", verbose=True)
    
    if not success:
        print("⚠️ API server might not have started correctly.")
        response = input("Continue and try to start Streamlit anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Give the API time to start
    print("⏳ Waiting for API server to start...")
    time.sleep(5)
    
    # Start Streamlit app
    print("\n🚀 Starting Streamlit app...")
    if platform.system() == "Windows":
        # For Windows, use proper quoting for paths with spaces
        streamlit_command = f"start \"\" \"{python_cmd}\" -m streamlit run streamlit_app.py"
    else:
        # For Unix systems
        streamlit_command = f"\"{python_cmd}\" -m streamlit run streamlit_app.py &"
    
    success = run_command(streamlit_command, "Starting Streamlit app", verbose=True)
    
    if not success:
        print("❌ Failed to start Streamlit app.")
        return False
    
    print("\n✅ Services started successfully!")
    print("\n📊 Access the services:")
    print("  - API: http://localhost:8002/docs")
    print("  - Streamlit: http://localhost:8501")
    
    print("\n⚠️ Press Ctrl+C in each terminal window to stop the services when done.")
    
    return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Real USA Agricultural Detection System")
    parser.add_argument("--env", "-e", choices=ENVIRONMENTS, default="development",
                        help="Environment to deploy to")
    parser.add_argument("--skip-checks", "-s", action="store_true",
                        help="Skip dependency checks")
    parser.add_argument("--no-pull", "-n", action="store_true",
                        help="Skip pulling latest images")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output for debugging")
    parser.add_argument("--local", "-l", action="store_true",
                        help="Run services locally without Docker")
    
    # Parse arguments and make verbose flag global so run_command can use it
    global VERBOSE
    args = parser.parse_args()
    VERBOSE = args.verbose
    
    args = parser.parse_args()
    
    print_header(f"Real USA Agricultural Detection System - {args.env.upper()} Deployment")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate test data for demo purposes
    generate_test_data()
    
    # Check if a docker-compose file exists, create minimal if not
    create_minimal_compose_file()
    
    # Local mode (without Docker)
    if args.local:
        run_services_locally(args.env)
        sys.exit(0)
    
    # Check system dependencies
    if not args.skip_checks and not check_dependencies():
        print("❌ Dependency check failed. Please install required dependencies.")
        print("💡 TIP: Run with --local flag to run services without Docker")
        sys.exit(1)
    
    # Check environment configuration
    if not check_environment(args.env):
        print("❌ Environment configuration check failed.")
        sys.exit(1)
    
    # Stop any running containers
    if not stop_running_containers():
        print("⚠️ Warning: Failed to stop existing containers.")
    
    # Pull latest images for production
    if not args.no_pull and not pull_latest_images(args.env):
        print("⚠️ Warning: Failed to pull latest images.")
    
    # Start services
    if not start_services(args.env):
        print("❌ Failed to start services.")
        print("💡 TIP: Run with --verbose flag for more details")
        print("💡 TIP: Run with --local flag to run services without Docker")
        print("\nPossible issues:")
        print("1. Docker service is not running - check Docker Desktop is started")
        print("2. Docker Compose isn't correctly installed")
        print("3. There's a conflict with existing containers")
        print("4. The docker-compose.yml file has syntax errors")
        print("\n➡️ To run without Docker:")
        print(f"   python deploy.py --env {args.env} --local")
        sys.exit(1)
    
    # Verify services are running
    verify_services()
    
    print_header("Deployment Complete")
    print("✅ System deployed successfully!")
    print("📊 Dashboard URLs:")
    print("  - Streamlit Dashboard: http://localhost:8501")
    print("  - API Documentation: http://localhost:8002/docs")
    print("  - Main Frontend: http://localhost:3000 (if frontend service is configured)")
    
    print("\n🔍 Deployment Summary:")
    print(f"  - Environment: {args.env}")
    print(f"  - Mode: {'Local (no Docker)' if args.local else 'Docker'}")
    print(f"  - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🛑 To stop the services:")
    if args.local:
        print("  - Press Ctrl+C in the terminal windows running the services")
        print("  - Or close the terminal windows")
    else:
        print("  - Run: docker-compose down")
    
    print("\n💡 For troubleshooting:")
    print("  - Check logs: docker-compose logs")
    print("  - Restart services: docker-compose restart")
    print("  - Try local mode: python deploy.py --env development --local")
    
if __name__ == "__main__":
    main()