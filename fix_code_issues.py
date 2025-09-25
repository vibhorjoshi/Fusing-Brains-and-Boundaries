"""
Code Quality Fix Utility

This script contains common fixes for various code quality issues that might
appear across the codebase, particularly in notebooks and scripts.
"""

import re
import os
from pathlib import Path

def fix_matplotlib_colormaps(file_path):
    """Fix matplotlib colormap references from plt.get_cmap('xxx') to plt.get_cmap('xxx')"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match plt.get_cmap('colormap_name')
    pattern = r'plt\.cm\.(\w+)'
    
    def replace_colormap(match):
        colormap_name = match.group(1)
        return f"plt.get_cmap('{colormap_name}')"
    
    updated_content = re.sub(pattern, replace_colormap, content)
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Fixed matplotlib colormaps in {file_path}")
        return True
    return False

def fix_opencv_issues(file_path):
    """Common OpenCV fixes"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content
    
    # Fix fillPoly calls
    updated_content = re.sub(
        r'cv2\.fillPoly\(([^,]+),\s*\[([^\]]+)\],\s*([^)]+)\)',
        r'cv2.fillPoly(\1, [np.array(np.array(\2, dtype=np.int32), dtype=np.int32)], \3)',
        updated_content
    )
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Fixed OpenCV issues in {file_path}")
        return True
    return False

def fix_shapely_geoms(file_path):
    """Fix Shapely geometry operations"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content
    
    # Fix .geoms access for MultiPolygons
    updated_content = re.sub(
        r'(\w+)\.geoms',
        r'getattr(\1, "geoms", [\1] if hasattr(\1, "coords") else [])',
        updated_content
    )
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Fixed Shapely geometry operations in {file_path}")
        return True
    return False

def fix_common_notebook_issues():
    """Common fixes for notebook-style code"""
    fixes = [
        ('!pip install', '%pip install'),
        ('plt.get_cmap('Set3')', "plt.get_cmap('Set3')"),
        ('plt.get_cmap('Pastel1')', "plt.get_cmap('Pastel1')"),
        ('plt.get_cmap('viridis')', "plt.get_cmap('viridis')"),
        ('plt.get_cmap('plasma')', "plt.get_cmap('plasma')"),
    ]
    
    print("Common notebook fixes:")
    for old, new in fixes:
        print(f"  Replace: {old} → {new}")

def scan_and_fix_directory(directory_path):
    """Scan a directory and fix common issues in Python files"""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist")
        return
    
    python_files = list(directory.rglob("*.py"))
    
    print(f"Scanning {len(python_files)} Python files in {directory_path}")
    
    fixes_applied = 0
    for file_path in python_files:
        try:
            if fix_matplotlib_colormaps(file_path):
                fixes_applied += 1
            if fix_opencv_issues(file_path):
                fixes_applied += 1
            if fix_shapely_geoms(file_path):
                fixes_applied += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Applied fixes to {fixes_applied} files")

if __name__ == "__main__":
    print("Code Quality Fix Utility")
    print("========================")
    
    # Fix common notebook issues reference
    fix_common_notebook_issues()
    
    # Scan current directory
    print(f"\nScanning current directory...")
    scan_and_fix_directory(".")
    
    print("\nFix utility completed!")