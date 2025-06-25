# pylint: skip-file

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # or wherever your module is


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyrite"
copyright = "2025, M.J. van der Lugt"
author = "M.J. van der Lugt"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "myst_nb",
]
# numpydoc_show_class_members = False


autosummary_generate = True  # <- crucial

templates_path = ["_templates"]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": False,
    "private-members": True,
}
autodoc_typehints = "none"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference",
        "https://docs.scipy.org/doc/scipy/objects.inv",
    ),
    "rdkit": ("https://www.rdkit.org/docs", None),
    "openff": ("https://docs.openforcefield.org/projects/toolkit/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
}

# turn off display of the source‐code download link
plot_html_show_source_link = False

# turn off all format download links (png, pdf, etc.)
plot_html_show_formats = False
