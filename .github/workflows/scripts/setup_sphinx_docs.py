import os
import sys
import shutil
from pathlib import Path
import datetime

def setup_sphinx_docs():
    """Set up Sphinx documentation structure"""
    # Create directory structure
    docs_dir = Path("docs")
    source_dir = docs_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Create conf.py
    conf_py = source_dir / "conf.py"
    if not conf_py.exists():
        with open(conf_py, "w") as f:
            f.write(f"""
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath('../..'))

project = 'GeoAI Research'
copyright = '{datetime.datetime.now().year}, Research Team'
author = 'GeoAI Research Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_default_options = {{
    'members': True,
    'show-inheritance': True,
}}

autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
""")
    
    # Create index.rst
    index_rst = source_dir / "index.rst"
    if not index_rst.exists():
        with open(index_rst, "w") as f:
            f.write("""Welcome to GeoAI Research Documentation
======================================

This is the API documentation for the GeoAI Research project.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   api
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
""")
    
    # Create api.rst
    api_rst = source_dir / "api.rst"
    if not api_rst.exists():
        with open(api_rst, "w") as f:
            f.write("""API Reference
============

This page contains the API reference for the main modules.

.. autosummary::
   :toctree: _autosummary
   :recursive:
   
   src
   backend
""")

if __name__ == "__main__":
    setup_sphinx_docs()