import os
import sys

# Add project root to sys.path if not already there
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the 2D architecture diagram function
from src.architecture_2d import save_2d_architecture_diagram

# Generate the diagram
output_path = save_2d_architecture_diagram()
print(f"Generated architecture diagram at: {output_path}")