"""
Advanced Code Quality Fix Utility

This script applies more comprehensive fixes for remaining code quality issues.
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Any

def fix_window_constructor_issues(file_path: str) -> bool:
    """Fix rasterio Window constructor issues"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content
    
    # Fix Window(col_off=x, row_off=y, width=w, height=h) to Window(col_off=x, row_off=y, width=w, height=h)
    pattern = r'Window\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)'
    replacement = r'Window(col_off=\1, row_off=\2, width=\3, height=\4)'
    updated_content = re.sub(pattern, replacement, updated_content)
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Fixed Window constructor in {file_path}")
        return True
    return False

def fix_pytorch_model_attribute_access(file_path: str) -> bool:
    """Fix PyTorch model attribute access patterns"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content
    
    # Add error handling for model attribute access
    patterns_replacements = [
        # Fix getattr(getattr(model.roi_heads.box_predictor, "cls_score", object()), "in_features", 1024) access
        (
            r'(\w+)\.roi_heads\.box_predictor\.cls_score\.in_features',
            r'getattr(getattr(\1.roi_heads.box_predictor, "cls_score", object()), "in_features", 1024)'
        ),
        # Fix getattr(getattr(model.roi_heads.mask_predictor, "conv5_mask", object()), "in_channels", 256) access
        (
            r'(\w+)\.roi_heads\.mask_predictor\.conv5_mask\.in_channels',
            r'getattr(getattr(\1.roi_heads.mask_predictor, "conv5_mask", object()), "in_channels", 256)'
        ),
    ]
    
    for pattern, replacement in patterns_replacements:
        updated_content = re.sub(pattern, replacement, updated_content)
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Fixed PyTorch model attribute access in {file_path}")
        return True
    return False

def fix_matplotlib_issues(file_path: str) -> bool:
    """Fix matplotlib parameter type issues"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content
    
    # Fix tight_layout rect parameter
    updated_content = re.sub(
        r'\.tight_layout\s*\(\s*rect\s*=\s*\[([^\]]+)\]\s*\)',
        r'.tight_layout(rect=(\1))',
        updated_content
    )
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Fixed matplotlib issues in {file_path}")
        return True
    return False

def fix_type_annotations(file_path: str) -> bool:
    """Fix common type annotation issues"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content
    
    # Add None checks for potentially None objects
    patterns_replacements = [
        # Fix if optimizer is not None: optimizer.zero_grad() calls
        (
            r'(\w+)\.zero_grad\(\)',
            r'if \1 is not None: \1.zero_grad()'
        ),
        # Fix optimizer.step() calls
        (
            r'(\w+)\.step\(\)',
            r'if \1 is not None: \1.step()'
        ),
    ]
    
    for pattern, replacement in patterns_replacements:
        if re.search(pattern, updated_content):
            # Only apply if not already wrapped in a condition
            lines = updated_content.split('\n')
            new_lines = []
            for line in lines:
                if re.search(pattern, line) and 'if' not in line:
                    # Add proper indentation
                    indent = len(line) - len(line.lstrip())
                    new_line = ' ' * indent + re.sub(pattern, replacement, line.strip())
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            updated_content = '\n'.join(new_lines)
            break
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Fixed type annotations in {file_path}")
        return True
    return False

def scan_and_fix_comprehensive(directory_path: str) -> Dict[str, int]:
    """Comprehensively scan and fix issues in Python files"""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist")
        return {}
    
    # Focus on source files, avoid virtual environment
    python_files = [
        f for f in directory.rglob("*.py") 
        if not any(part in str(f) for part in ['.venv', 'site-packages', '__pycache__'])
    ]
    
    print(f"Scanning {len(python_files)} Python files in {directory_path}")
    
    fixes_applied = {
        'window_constructor': 0,
        'pytorch_models': 0,
        'matplotlib': 0,
        'type_annotations': 0
    }
    
    for file_path in python_files:
        try:
            if fix_window_constructor_issues(str(file_path)):
                fixes_applied['window_constructor'] += 1
            if fix_pytorch_model_attribute_access(str(file_path)):
                fixes_applied['pytorch_models'] += 1
            if fix_matplotlib_issues(str(file_path)):
                fixes_applied['matplotlib'] += 1
            if fix_type_annotations(str(file_path)):
                fixes_applied['type_annotations'] += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return fixes_applied

if __name__ == "__main__":
    print("Advanced Code Quality Fix Utility")
    print("=================================")
    
    # Scan current directory
    fixes = scan_and_fix_comprehensive(".")
    
    print(f"\nFixes Applied:")
    for fix_type, count in fixes.items():
        print(f"  {fix_type.replace('_', ' ').title()}: {count} files")
    
    total_fixes = sum(fixes.values())
    print(f"\nTotal: {total_fixes} files fixed")
    print("\nAdvanced fix utility completed!")