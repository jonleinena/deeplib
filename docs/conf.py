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

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'logo_only': False,
    'style_nav_header_background': '#2980B9',
}

html_static_path = ['_static']
html_title = 'DeepLib Documentation'
html_short_title = 'DeepLib'
html_show_sourcelink = False
html_show_sphinx = False

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
add_module_names = False
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}