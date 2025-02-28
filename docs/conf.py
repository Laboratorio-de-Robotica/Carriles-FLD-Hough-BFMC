# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Picking U2Net'
copyright = '2025, Alejandro Silvestri'
author = 'Alejandro Silvestri'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../'))  # ruta el código, relativa a este archivo conf.py
sys.path.insert(0, os.path.abspath('../lib'))

extensions = [
    'sphinx.ext.autodoc', # documentación automática, extrae la documentación de los archivos del proyecto
    'sphinx.ext.napoleon',  # interpreta docstrings estilo Google
    'myst_parser', # soporte para Markdown
    'sphinx_rtd_theme' # tema Read The Docs
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'es'

# Tema Read The Docs
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Lista de módulos que no estarán disponibles al documentar
autodoc_mock_imports = ["torch"]
html_show_sourcelink = False