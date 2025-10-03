#!/usr/bin/env python3
"""
GeoAI Research - File System Analyzer
Analyzes the directory structure and file types in the project
"""

import os
import sys
import datetime
import re
from collections import defaultdict

def main():
    """Main function"""
    print("GeoAI Research - File System Analyzer")
    print("-------------------------------------")
    
    # Define the root directory
    root_dir = os.getcwd()
    print(f"Analyzing directory: {root_dir}")
    print(f"Current time: {datetime.datetime.now()}")
    print()
    
    # Analyze directory structure
    file_types = defaultdict(int)
    dir_structure = {}
    total_size = 0
    largest_files = []
    
    # Walk through directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip __pycache__ directories
        if "__pycache__" in dirpath:
            continue
            
        # Calculate relative path
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == ".":
            rel_path = ""
            
        # Store directory info
        dir_structure[rel_path] = {
            "file_count": len(filenames),
            "dir_count": len(dirnames),
            "size": 0
        }
        
        # Process files
        for filename in filenames:
            # Get file extension
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            file_types[ext] += 1
            
            # Get file size
            file_path = os.path.join(dirpath, filename)
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                dir_structure[rel_path]["size"] += file_size
                
                # Track largest files
                largest_files.append((file_path, file_size))
                largest_files.sort(key=lambda x: x[1], reverse=True)
                largest_files = largest_files[:10]  # Keep only top 10
            except:
                pass
    
    # Print file type statistics
    print("=== File Type Statistics ===")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{ext or 'No extension'}: {count} files")
    print()
    
    # Print top-level directory statistics
    print("=== Top-Level Directory Statistics ===")
    for dirname, dirinfo in sorted(dir_structure.items()):
        if dirname and "/" not in dirname and "\\" not in dirname:
            print(f"{dirname or 'Root'}: {dirinfo['file_count']} files, {dirinfo['dir_count']} subdirs, {format_size(dirinfo['size'])}")
    print()
    
    # Print largest files
    print("=== Largest Files ===")
    for file_path, size in largest_files:
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"{rel_path}: {format_size(size)}")
    print()
    
    # Look for Python scripts
    python_scripts = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, root_dir)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Extract imports
                        imports = re.findall(r'import\s+([a-zA-Z0-9_,\s.]+)|from\s+([a-zA-Z0-9_.]+)\s+import', content)
                        flat_imports = []
                        for imp in imports:
                            if imp[0]:  # import x
                                modules = [m.strip() for m in imp[0].split(',')]
                                flat_imports.extend(modules)
                            elif imp[1]:  # from x import
                                flat_imports.append(imp[1])
                        
                        # Count lines
                        lines = content.count('\n') + 1
                        python_scripts.append((rel_path, lines, flat_imports))
                except:
                    pass
    
    # Print Python script statistics
    print("=== Python Script Statistics ===")
    python_scripts.sort(key=lambda x: x[1], reverse=True)
    for script, lines, imports in python_scripts[:10]:  # Top 10 by line count
        print(f"{script}: {lines} lines, {len(imports)} imports")
    print()
    
    # Print common imports
    all_imports = []
    for _, _, imports in python_scripts:
        all_imports.extend(imports)
    
    import_counts = defaultdict(int)
    for imp in all_imports:
        base_module = imp.split('.')[0]
        import_counts[base_module] += 1
    
    print("=== Common Python Imports ===")
    for module, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"{module}: used in {count} places")
    
    # Create a summary file
    output_file = "filesystem_analysis.txt"
    try:
        with open(output_file, "w") as f:
            f.write("GeoAI Research - File System Analysis\n")
            f.write("-------------------------------------\n")
            f.write(f"Generated on: {datetime.datetime.now()}\n")
            f.write(f"Total files: {sum(file_types.values())}\n")
            f.write(f"Total directories: {len(dir_structure)}\n")
            f.write(f"Total size: {format_size(total_size)}\n\n")
            
            f.write("=== File Type Distribution ===\n")
            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{ext or 'No extension'}: {count} files\n")
            
            f.write("\n=== Python Script Analysis ===\n")
            for script, lines, imports in python_scripts:
                f.write(f"{script}: {lines} lines, {len(imports)} imports\n")
                if imports:
                    f.write(f"  Imports: {', '.join(sorted(set([i.split('.')[0] for i in imports])))}\n")
            
            f.write("\n=== Common Python Imports ===\n")
            for module, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{module}: used in {count} places\n")
        
        print(f"\nCreated analysis file: {output_file}")
        return 0
    except Exception as e:
        print(f"Error creating analysis file: {e}")
        return 1


def format_size(size_bytes):
    """Format size in bytes to human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


if __name__ == "__main__":
    sys.exit(main())