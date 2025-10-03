#!/usr/bin/env python3
"""
CI/CD Pipeline Verification Script
Checks that the basic requirements for running the CI/CD pipeline are in place
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")

def check_environment():
    print_section("Python Environment")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Working Directory: {os.getcwd()}")

def check_workflow_files():
    print_section("GitHub Workflow Files")
    workflow_dir = Path(".github/workflows")
    
    if not workflow_dir.exists():
        print(f"❌ Workflow directory not found: {workflow_dir}")
        return False
    
    workflow_files = list(workflow_dir.glob("*.yml"))
    if not workflow_files:
        print("❌ No workflow files found")
        return False
    
    print(f"✅ Found {len(workflow_files)} workflow files:")
    for wf in workflow_files:
        print(f"  - {wf}")
    
    # Check for the enhanced CI/CD pipeline
    enhanced_ci_file = workflow_dir / "enhanced-ci-cd-pipeline.yml"
    if enhanced_ci_file.exists():
        print(f"✅ Found enhanced CI/CD pipeline file: {enhanced_ci_file}")
    else:
        print(f"❌ Enhanced CI/CD pipeline file not found: {enhanced_ci_file}")
        return False
    
    return True

def check_files():
    print_section("Required Project Files")
    required_files = [
        "requirements.txt",
        "requirements_ci.txt",
        "setup.py",
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ Found {file}")
        else:
            print(f"❌ Missing {file}")
    
    return all(Path(file).exists() for file in required_files)

def try_import():
    print_section("Import Checks")
    basic_imports = ["os", "sys", "json", "subprocess", "pathlib"]
    for module in basic_imports:
        try:
            __import__(module)
            print(f"✅ Successfully imported {module}")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
    
    # Try optional imports for our project
    optional_imports = ["numpy", "pandas", "matplotlib"]
    for module in optional_imports:
        try:
            __import__(module)
            print(f"✅ Successfully imported {module}")
        except ImportError as e:
            print(f"⚠️ Optional module {module} not available: {e}")

def run_security_check():
    print_section("Security Check Script Test")
    security_check = Path("security_check.py")
    
    if not security_check.exists():
        print(f"❌ Security check script not found: {security_check}")
        return False
    
    print(f"✅ Found security check script: {security_check}")
    print("ℹ️ Not executing security checks as dependencies may be missing")
    return True

def main():
    print("\n🚀 CI/CD Pipeline Verification")
    print("=" * 40)
    
    checks = [
        check_environment(),
        check_workflow_files(),
        check_files(),
        try_import(),
        run_security_check()
    ]
    
    # Filter out None values
    checks = [c for c in checks if c is not None]
    
    if all(checks):
        print("\n✅ All verification checks passed! CI/CD pipeline should work.")
        print("Run the CI/CD pipeline with: ./run_cicd.ps1 or ./run_cicd.sh")
        return 0
    else:
        print("\n⚠️ Some verification checks failed. Fix the issues before running the CI/CD pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main())