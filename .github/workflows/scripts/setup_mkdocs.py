import os
from pathlib import Path
import yaml

def setup_mkdocs():
    """Set up MkDocs documentation structure"""
    # Create MkDocs configuration if it doesn't exist
    if not os.path.exists("mkdocs.yml"):
        mkdocs_config = {
            "site_name": "GeoAI Research Documentation",
            "site_description": "Documentation for GeoAI Research project",
            "site_author": "GeoAI Research Team",
            "repo_url": "https://github.com/vibhorjoshi/Fusing-Brains-and-Boundaries",
            "nav": [
                {"Home": "index.md"},
                {"User Guide": [
                    {"Installation": "guide/installation.md"},
                    {"Usage": "guide/usage.md"},
                ]},
                {"API Reference": "api/index.md"},
                {"Development": [
                    {"Contributing": "development/contributing.md"},
                    {"Architecture": "development/architecture.md"},
                ]},
                {"About": [
                    {"Release Notes": "about/release-notes.md"},
                    {"License": "about/license.md"},
                ]},
            ],
            "theme": {
                "name": "material",
                "palette": {
                    "primary": "blue",
                    "accent": "blue",
                },
                "features": [
                    "navigation.tabs",
                    "navigation.instant",
                    "navigation.tracking",
                    "content.code.annotate",
                    "content.tabs.link",
                ],
            },
            "markdown_extensions": [
                "pymdownx.highlight",
                "pymdownx.superfences",
                "pymdownx.inlinehilite",
                "pymdownx.tabbed",
                "pymdownx.arithmatex",
                "admonition",
                {"toc": {"permalink": True}},
            ],
            "plugins": [
                "search",
                "mkdocstrings",
            ],
        }
        
        with open("mkdocs.yml", "w") as f:
            yaml.dump(mkdocs_config, f, default_flow_style=False)
    
    # Create directory structure
    docs_dir = Path("docs/docs")
    for subdir in ["guide", "api", "development", "about"]:
        (docs_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create index.md
    index_md = docs_dir / "index.md"
    if not index_md.exists():
        with open(index_md, "w") as f:
            f.write("""# GeoAI Research Documentation

Welcome to the GeoAI Research documentation. This documentation provides information about 
the project, how to use it, and API reference.

## Overview

The GeoAI Research project is focused on real USA agricultural detection using machine learning 
and computer vision techniques. The system uses advanced deep learning models to identify and 
analyze agricultural areas from satellite imagery.

## Getting Started

To get started with the GeoAI Research system:

1. Check the [Installation Guide](guide/installation.md) for setup instructions
2. Review the [Usage Guide](guide/usage.md) for how to use the system
3. Explore the [API Reference](api/index.md) for detailed technical information

## Features

- Building footprint detection and analysis
- Crop field identification
- Agricultural area measurement
- Real-time performance monitoring
- Interactive visualizations
""")
    
    # Create installation guide
    install_md = docs_dir / "guide/installation.md"
    if not install_md.exists():
        with open(install_md, "w") as f:
            f.write("""# Installation Guide

This guide explains how to install and set up the GeoAI Research system.

## Prerequisites

- Python 3.11 or higher
- Docker (optional, for containerized deployment)
- CUDA-compatible GPU (recommended for optimal performance)

## Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/vibhorjoshi/Fusing-Brains-and-Boundaries.git
cd Fusing-Brains-and-Boundaries
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Docker deployment (optional)

```bash
docker-compose up -d
```

### 4. Local deployment

```bash
python deploy_local.py --env development
```
""")

if __name__ == "__main__":
    setup_mkdocs()