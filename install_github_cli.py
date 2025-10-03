#!/usr/bin/env python3
"""
GitHub CLI Installer Script
Downloads and installs GitHub CLI locally using Python
"""

import os
import sys
import platform
import subprocess
import zipfile
import shutil
import tempfile
import urllib.request
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

def print_step(step):
    print(f"\n{Colors.BLUE}[STEP] {step}{Colors.END}")

def print_success(message):
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}[ERROR] {message}{Colors.END}")

def print_info(message):
    print(f"{Colors.YELLOW}[INFO] {message}{Colors.END}")

def get_system_info():
    """Get system information"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Map architecture
    if machine in ["x86_64", "amd64"]:
        arch = "amd64"
    elif machine in ["i386", "i686", "x86"]:
        arch = "386"
    elif machine in ["arm64", "aarch64"]:
        arch = "arm64"
    elif machine.startswith("arm"):
        arch = "arm"
    else:
        arch = machine
        
    return system, arch

def download_github_cli():
    """Download GitHub CLI release for the current platform"""
    system, arch = get_system_info()
    
    print_step(f"Downloading GitHub CLI for {system} {arch}")
    
    # Get the latest release URL based on platform
    if system == "windows":
        # For Windows, we'll get the ZIP version
        download_url = f"https://github.com/cli/cli/releases/download/v2.43.1/gh_2.43.1_windows_{arch}.zip"
        file_name = "gh_windows.zip"
    elif system == "darwin":  # macOS
        download_url = f"https://github.com/cli/cli/releases/download/v2.43.1/gh_2.43.1_macOS_{arch}.tar.gz"
        file_name = "gh_macos.tar.gz"
    elif system == "linux":
        download_url = f"https://github.com/cli/cli/releases/download/v2.43.1/gh_2.43.1_linux_{arch}.tar.gz"
        file_name = "gh_linux.tar.gz"
    else:
        print_error(f"Unsupported system: {system}")
        return None
    
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    download_path = os.path.join(temp_dir, file_name)
    
    # Download the file
    print_info(f"Downloading from: {download_url}")
    try:
        urllib.request.urlretrieve(download_url, download_path)
        print_success(f"Downloaded to: {download_path}")
        return download_path, temp_dir
    except Exception as e:
        print_error(f"Failed to download: {str(e)}")
        return None, temp_dir

def extract_github_cli(download_path, temp_dir, install_dir):
    """Extract the downloaded GitHub CLI archive"""
    print_step("Extracting GitHub CLI")
    
    system, _ = get_system_info()
    
    try:
        if system == "windows":
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        else:
            # For macOS and Linux
            import tarfile
            with tarfile.open(download_path, 'r:gz') as tar_ref:
                tar_ref.extractall(temp_dir)
        
        print_success("Extraction completed")
        
        # Find the bin directory with gh executable
        for root, dirs, files in os.walk(temp_dir):
            if system == "windows":
                if "gh.exe" in files:
                    src_path = os.path.join(root)
                    break
            else:
                if "gh" in files and not os.path.islink(os.path.join(root, "gh")):
                    src_path = os.path.join(root)
                    break
        else:
            print_error("Could not find GitHub CLI executable in the archive")
            return False
            
        # Ensure install directory exists
        os.makedirs(install_dir, exist_ok=True)
        
        # Copy executable and related files to install directory
        for item in os.listdir(src_path):
            s = os.path.join(src_path, item)
            d = os.path.join(install_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
                
        print_success(f"GitHub CLI installed to: {install_dir}")
        return True
    
    except Exception as e:
        print_error(f"Extraction failed: {str(e)}")
        return False

def add_to_path(install_dir):
    """Add the install directory to PATH temporarily"""
    print_step("Adding GitHub CLI to PATH")
    
    # Add the directory to the current process PATH
    os.environ["PATH"] = f"{install_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    
    print_info("GitHub CLI added to PATH for this session")
    print_info("To make this permanent, add this directory to your system PATH:")
    print_info(f"  {install_dir}")
    
    # For Windows, also provide PowerShell instruction
    system, _ = get_system_info()
    if system == "windows":
        ps_cmd = f'$env:PATH = "{install_dir};" + $env:PATH'
        print_info("For PowerShell, you can use:")
        print_info(f"  {ps_cmd}")
    
    return True

def verify_installation(install_dir):
    """Verify GitHub CLI installation"""
    print_step("Verifying GitHub CLI installation")
    
    system, _ = get_system_info()
    if system == "windows":
        gh_path = os.path.join(install_dir, "gh.exe")
    else:
        gh_path = os.path.join(install_dir, "gh")
    
    if not os.path.exists(gh_path):
        print_error(f"GitHub CLI executable not found at: {gh_path}")
        return False
    
    # Try running gh version
    try:
        result = subprocess.run([gh_path, "--version"], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        print_success(f"GitHub CLI installed successfully:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to run GitHub CLI: {str(e)}")
        print_error(f"Output: {e.stdout}")
        print_error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print_error(f"Error verifying installation: {str(e)}")
        return False

def authenticate_github_cli(install_dir):
    """Guide to authenticate GitHub CLI"""
    print_step("GitHub CLI Authentication")
    
    system, _ = get_system_info()
    if system == "windows":
        gh_path = os.path.join(install_dir, "gh.exe")
    else:
        gh_path = os.path.join(install_dir, "gh")
    
    print_info("To authenticate GitHub CLI, you need to run:")
    print_info(f"  {gh_path} auth login")
    print_info("\nFollow the interactive prompts to complete authentication.")
    print_info("You can choose to authenticate with:")
    print_info("  1. GitHub.com account via web browser")
    print_info("  2. GitHub.com account via token")
    print_info("  3. GitHub Enterprise Server account")
    
    choice = input("\nDo you want to authenticate now? (y/n): ")
    if choice.lower() == 'y':
        try:
            subprocess.run([gh_path, "auth", "login"], check=True)
            print_success("Authentication process completed.")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Authentication failed: {str(e)}")
            return False
    else:
        print_info("You can authenticate later by running the command manually.")
        return False

def run_workflow(install_dir):
    """Run the GitHub workflow"""
    print_step("Running GitHub Workflow")
    
    system, _ = get_system_info()
    if system == "windows":
        gh_path = os.path.join(install_dir, "gh.exe")
    else:
        gh_path = os.path.join(install_dir, "gh")
    
    workflow_name = "🌾 Real USA Agricultural Detection CI/CD Pipeline"
    
    print_info(f"Running workflow: {workflow_name}")
    print_info("Branch: main")
    print_info("Environment: development")
    
    try:
        cmd = [
            gh_path, "workflow", "run",
            workflow_name,
            "--ref", "main",
            "--field", "environment=development"
        ]
        subprocess.run(cmd, check=True)
        print_success("Workflow run triggered successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to run workflow: {str(e)}")
        return False

def main():
    print(f"\n{Colors.BOLD}GitHub CLI Installer and Workflow Runner{Colors.END}")
    print("This script will download, install, and run GitHub CLI locally.\n")
    
    # Define install directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    install_dir = os.path.join(script_dir, "gh_cli")
    
    # Download GitHub CLI
    download_path, temp_dir = download_github_cli()
    if not download_path:
        print_error("Download failed. Exiting.")
        return 1
    
    # Extract and install
    if not extract_github_cli(download_path, temp_dir, install_dir):
        print_error("Installation failed. Exiting.")
        return 1
    
    # Add to PATH
    add_to_path(install_dir)
    
    # Verify installation
    if not verify_installation(install_dir):
        print_error("Verification failed. Exiting.")
        return 1
    
    # Authenticate
    auth_success = authenticate_github_cli(install_dir)
    
    # Run workflow if authenticated
    if auth_success:
        run_workflow(install_dir)
    else:
        print_info("Authentication required before running workflow.")
        print_info(f"1. Authenticate: {os.path.join(install_dir, 'gh')} auth login")
        print_info(f"2. Run workflow: {os.path.join(install_dir, 'gh')} workflow run \"🌾 Real USA Agricultural Detection CI/CD Pipeline\" --ref main --field environment=development")
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    print_success("GitHub CLI setup completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())