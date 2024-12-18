# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'DeepLib'
copyright = '2024, Jon Leiñena Otamendi'
author = 'Jon Leiñena Otamendi'
version = '1.1.0'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',    # Core extension for API documentation
    'sphinx.ext.viewcode',   # Add links to highlighted source code
    'sphinx.ext.napoleon',   # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',# Link to other project's documentation
    'myst_parser'           # Support for Markdown files
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'en'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Use the Read the Docs theme
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- autodoc configuration -------------------------------------------------
autodoc_member_order = 'bysource'  # Document members in source code order
add_module_names = False           # Don't prefix members with module name 