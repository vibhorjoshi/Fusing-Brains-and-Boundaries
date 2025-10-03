#!/usr/bin/env python3
"""
CI/CD Pipeline Fix Script
Fixes common issues in GitHub Actions workflow files
"""

import os
import sys
import re
from pathlib import Path

def fix_artifact_version(workflow_path):
    """Update deprecated actions/upload-artifact from v3 to v4"""
    print(f"Fixing actions/upload-artifact version in {workflow_path}")
    
    try:
        with open(workflow_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(workflow_path, 'r', encoding='utf-8-sig') as file:
            content = file.read()
    
    # Replace actions/upload-artifact@v3 with v4
    updated_content = content.replace('actions/upload-artifact@v3', 'actions/upload-artifact@v4')
    
    # Count replacements
    replacements = content.count('actions/upload-artifact@v3')
    
    try:
        with open(workflow_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
    except Exception as e:
        print(f"Error writing to file: {e}")
        return 0
    
    print(f"✓ Updated {replacements} occurrences of actions/upload-artifact@v3 to v4")
    return replacements

def check_required_files():
    """Check and create missing required files"""
    print("Checking for missing required files...")
    
    # List of required files and their template content
    required_files = {
        "streamlit_requirements.txt": """# Streamlit requirements file
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
geopandas>=0.13.2
folium>=0.14.0
branca>=0.6.0
rasterio>=1.3.8
shapely>=2.0.1
scikit-learn>=1.3.0
pillow>=10.0.0
requests>=2.31.0
watchdog>=3.0.0
pydeck>=0.8.0
altair>=5.0.0
bokeh>=3.3.0
scipy>=1.11.0
"""
    }
    
    created_files = []
    for filename, content in required_files.items():
        if not os.path.exists(filename):
            print(f"Creating missing file: {filename}")
            with open(filename, 'w') as file:
                file.write(content)
            created_files.append(filename)
    
    if created_files:
        print(f"✓ Created {len(created_files)} missing files: {', '.join(created_files)}")
    else:
        print("✓ All required files already exist")
    
    return created_files

def main():
    """Main function"""
    print("\n== CI/CD Pipeline Fix Script ==\n")
    
    # Fix actions/upload-artifact version
    workflow_path = ".github/workflows/enhanced-ci-cd-pipeline.yml"
    if not os.path.exists(workflow_path):
        print(f"❌ Workflow file not found: {workflow_path}")
        return 1
    
    fix_artifact_version(workflow_path)
    
    # Check and create missing required files
    created_files = check_required_files()
    
    print("\n✓ All fixes applied successfully!")
    
    # Provide git commands to commit the changes
    if created_files or workflow_path:
        print("\nTo commit these changes to your repository:")
        print("  git add", end=" ")
        if workflow_path:
            print(workflow_path, end=" ")
        for file in created_files:
            print(file, end=" ")
        print("\n  git commit -m \"Fix CI/CD pipeline issues\"")
        print("  git push origin main")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())