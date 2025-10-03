import os
import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("deploy")

# Add the src directory to the path so we can import config_manager
src_dir = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_dir))

try:
    from config_manager import set_environment, get_config
except ImportError:
    logger.error("Could not import config_manager. Make sure the src directory is in the Python path.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy the GeoAI Research application")
    parser.add_argument(
        "--env", "-e",
        default=os.environ.get("ENVIRONMENT", "development"),
        choices=["development", "staging", "production"],
        help="Target environment (default: development)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def setup_environment(env):
    """Set up the environment"""
    logger.info(f"Setting up environment: {env}")
    
    # Set the environment in the config manager
    set_environment(env)
    
    # Get the environment config
    config = get_config()
    logger.info(f"Loaded configuration for environment: {env}")
    
    # Set environment variables based on config
    os.environ["ENVIRONMENT"] = env
    os.environ["DEBUG"] = str(config["debug"]).lower()
    os.environ["LOG_LEVEL"] = config["log_level"]
    
    # Set database environment variables
    db_config = config["services"]["database"]
    os.environ["DB_HOST"] = db_config["host"]
    os.environ["DB_PORT"] = str(db_config["port"])
    os.environ["DB_NAME"] = db_config["name"]
    os.environ["DB_USER"] = db_config["user"]
    
    # Set API environment variables
    api_config = config["services"]["api"]
    os.environ["API_HOST"] = api_config["host"]
    os.environ["API_PORT"] = str(api_config["port"])
    os.environ["API_WORKERS"] = str(api_config["workers"])
    
    # Set model environment variables
    model_config = config["model"]
    os.environ["MODEL_PATH"] = model_config["path"]
    os.environ["MODEL_PRECISION"] = model_config["precision"]
    os.environ["MODEL_BATCH_SIZE"] = str(model_config["batch_size"])
    
    # Create environment-specific docker-compose override
    create_docker_compose_override(config)
    
    return config

def create_docker_compose_override(config):
    """Create a docker-compose.override.yml file based on the environment config"""
    services = {}
    
    # Configure API service
    services["api"] = {
        "environment": [
            f"ENVIRONMENT={os.environ['ENVIRONMENT']}",
            f"DEBUG={os.environ['DEBUG']}",
            f"LOG_LEVEL={os.environ['LOG_LEVEL']}",
            f"DB_HOST={os.environ['DB_HOST']}",
            f"DB_PORT={os.environ['DB_PORT']}",
            f"DB_NAME={os.environ['DB_NAME']}",
            f"DB_USER={os.environ['DB_USER']}",
            f"MODEL_PATH={os.environ['MODEL_PATH']}",
            f"MODEL_PRECISION={os.environ['MODEL_PRECISION']}",
            f"MODEL_BATCH_SIZE={os.environ['MODEL_BATCH_SIZE']}"
        ],
        "ports": [
            f"{config['services']['api']['port']}:{config['services']['api']['port']}"
        ]
    }
    
    # Configure number of workers based on environment
    services["api"]["deploy"] = {
        "replicas": config["services"]["api"]["workers"]
    }
    
    # Configure storage service if using S3
    if config["services"]["storage"]["type"] == "s3":
        services["storage"] = {
            "environment": [
                "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}",
                "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}",
                f"S3_BUCKET={config['services']['storage']['bucket']}",
                f"S3_REGION={config['services']['storage']['region']}"
            ]
        }
    
    # Create the override file
    docker_compose_override = {
        "version": "3.8",
        "services": services
    }
    
    with open("docker-compose.override.yml", "w") as f:
        json.dump(docker_compose_override, f, indent=2)
    
    logger.info("Created environment-specific docker-compose.override.yml")

def deploy_docker():
    """Deploy using Docker Compose"""
    logger.info("Deploying with Docker Compose")
    
    # Build the Docker images
    subprocess.run(["docker-compose", "build"], check=True)
    
    # Start the services
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    
    logger.info("Deployment completed successfully")
    return True

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Set up the environment
    config = setup_environment(args.env)
    
    # Deploy
    result = deploy_docker()
    
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()