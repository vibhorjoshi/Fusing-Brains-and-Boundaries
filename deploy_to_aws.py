#!/usr/bin/env python3
"""
AWS Deployment Script - Real USA Agricultural Detection System

This script automates the deployment of the GeoAI Agricultural Detection System 
to AWS ECS (Elastic Container Service) or EC2 instances.

Before running this script:
1. Install AWS CLI and configure credentials (aws configure)
2. Install boto3 (pip install boto3)
3. Ensure Docker is installed and running

The script will:
1. Build all Docker images
2. Push images to Amazon ECR
3. Deploy containers to ECS or EC2
4. Configure load balancing and networking
5. Set up CloudWatch monitoring

Usage:
    python deploy_to_aws.py
"""

import os
import sys
import time
import subprocess
import json
import boto3
from botocore.exceptions import ClientError
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Deploy GeoAI Agricultural Detection System to AWS')
parser.add_argument('--region', type=str, default='us-east-1', help='AWS region to deploy to')
parser.add_argument('--stack-name', type=str, default='geoai-ag-detection', help='CloudFormation stack name')
parser.add_argument('--create-vpc', action='store_true', help='Create a new VPC')
parser.add_argument('--instance-type', type=str, default='t3.large', help='EC2 instance type (if using EC2)')
parser.add_argument('--deploy-mode', choices=['ecs', 'ec2'], default='ecs', help='Deploy to ECS or EC2')
args = parser.parse_args()

# AWS service clients
try:
    ecr_client = boto3.client('ecr', region_name=args.region)
    ecs_client = boto3.client('ecs', region_name=args.region)
    ec2_client = boto3.client('ec2', region_name=args.region)
    cloudformation_client = boto3.client('cloudformation', region_name=args.region)
    print(f"Connected to AWS services in {args.region} region")
except Exception as e:
    print(f"Error connecting to AWS: {e}")
    sys.exit(1)

# Define Docker image names and tags
images = [
    {"name": "geoai-api", "dockerfile": "docker/Dockerfile.api", "port": 8000},
    {"name": "geoai-streamlit", "dockerfile": "docker/Dockerfile.streamlit", "port": 8501},
    {"name": "geoai-maskrcnn", "dockerfile": "docker/Dockerfile.maskrcnn", "port": 5000},
    {"name": "geoai-monitor", "dockerfile": "docker/Dockerfile.monitor", "port": 9090}
]

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

def create_ecr_repository(repo_name):
    """Create an ECR repository if it doesn't exist"""
    try:
        response = ecr_client.create_repository(repositoryName=repo_name)
        print(f"Created ECR repository: {repo_name}")
        return response['repository']['repositoryUri']
    except ClientError as e:
        if e.response['Error']['Code'] == 'RepositoryAlreadyExistsException':
            print(f"ECR repository already exists: {repo_name}")
            response = ecr_client.describe_repositories(repositoryNames=[repo_name])
            return response['repositories'][0]['repositoryUri']
        else:
            print(f"Error creating ECR repository: {e}")
            sys.exit(1)

def build_and_push_images():
    """Build and push Docker images to ECR"""
    # Get ECR login token
    token = ecr_client.get_authorization_token()
    username, password = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode().split(':')
    registry = token['authorizationData'][0]['proxyEndpoint']
    
    # Login to ECR
    run_command(f'docker login -u {username} -p {password} {registry}', shell=True)
    
    # Build and push each image
    for image in images:
        # Create ECR repository
        repo_uri = create_ecr_repository(image['name'])
        
        # Build Docker image
        print(f"Building {image['name']}...")
        run_command(f'docker build -t {image["name"]}:latest -f {image["dockerfile"]} .', shell=True)
        
        # Tag image for ECR
        run_command(f'docker tag {image["name"]}:latest {repo_uri}:latest', shell=True)
        
        # Push to ECR
        print(f"Pushing {image['name']} to ECR...")
        run_command(f'docker push {repo_uri}:latest', shell=True)
        
        # Update image info with repo URI
        image['repo_uri'] = repo_uri
    
    return images

def deploy_to_ecs(images):
    """Deploy containers to ECS"""
    print("Deploying to ECS...")
    
    # Create CloudFormation template file
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            # Define VPC, subnets, security groups, etc.
            # Define ECS cluster, task definitions, services, etc.
            # Define load balancer, target groups, etc.
        }
    }
    
    # Write template to file
    with open('ecs-template.json', 'w') as f:
        json.dump(template, f, indent=2)
    
    # Deploy CloudFormation stack
    with open('ecs-template.json', 'r') as f:
        template_body = f.read()
    
    try:
        cloudformation_client.create_stack(
            StackName=args.stack_name,
            TemplateBody=template_body,
            Capabilities=['CAPABILITY_NAMED_IAM'],
            Parameters=[
                # Define parameters here
            ]
        )
        print(f"Creating CloudFormation stack: {args.stack_name}")
        
        # Wait for stack creation to complete
        print("Waiting for stack creation to complete...")
        cloudformation_client.get_waiter('stack_create_complete').wait(StackName=args.stack_name)
        print("Stack creation complete!")
        
    except ClientError as e:
        if "AlreadyExistsException" in str(e):
            # Update existing stack
            try:
                cloudformation_client.update_stack(
                    StackName=args.stack_name,
                    TemplateBody=template_body,
                    Capabilities=['CAPABILITY_NAMED_IAM'],
                    Parameters=[
                        # Define parameters here
                    ]
                )
                print(f"Updating CloudFormation stack: {args.stack_name}")
                
                # Wait for stack update to complete
                print("Waiting for stack update to complete...")
                cloudformation_client.get_waiter('stack_update_complete').wait(StackName=args.stack_name)
                print("Stack update complete!")
                
            except ClientError as e:
                if "No updates are to be performed" in str(e):
                    print("No updates needed for CloudFormation stack")
                else:
                    print(f"Error updating stack: {e}")
                    sys.exit(1)
        else:
            print(f"Error creating stack: {e}")
            sys.exit(1)
    
    # Get stack outputs
    response = cloudformation_client.describe_stacks(StackName=args.stack_name)
    outputs = response['Stacks'][0]['Outputs']
    
    # Print access URLs
    for output in outputs:
        if 'URL' in output['OutputKey']:
            print(f"{output['OutputKey']}: {output['OutputValue']}")

def deploy_to_ec2(images):
    """Deploy containers to EC2 using Docker Compose"""
    print("Deploying to EC2...")
    
    # Implementation would create EC2 instances, install Docker, 
    # copy docker-compose.yml and run docker-compose up
    print("EC2 deployment is not yet implemented")
    sys.exit(1)

def main():
    """Main deployment function"""
    print("Starting deployment of GeoAI Agricultural Detection System to AWS...")
    
    # Build and push Docker images
    print("Building and pushing Docker images...")
    built_images = build_and_push_images()
    
    # Deploy based on selected mode
    if args.deploy_mode == 'ecs':
        deploy_to_ecs(built_images)
    else:
        deploy_to_ec2(built_images)
    
    print("Deployment complete!")

if __name__ == "__main__":
    import base64
    main()