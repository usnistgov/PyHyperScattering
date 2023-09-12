# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))


# -- Project information -----------------------------------------------------

project = "PyHyperScattering"
copyright = (
    ": Official Contribution of the US Government.  Not subject to copyright in the United States."
)
author = "Peter Beaucage"

# The full version, including alpha/beta/rc tags
from PyHyperScattering import __version__

release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",  # numpydoc and google docstrings
]

# Ignore annoying type exception warnings which often come from newlines
nitpick_ignore = [("py:class", "type")]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

html_theme = "pydata_sphinx_theme"
html_logo = "source/_static/Logo_PyHyperO9_Light.svg"
html_theme_options = {
    "logo": {
        "image_light": "source/_images/Logo_PyHyperO9_Light.svg",
        "image_dark": "source/_images/Logo_PyHyperO10_Dark.svg",
    },
    "github_url": "https://github.com/usnistgov/PyHyperScattering",
    "collapse_navigation": True,
    #   "external_links": [
    #       {"name": "Learn", "url": "https://numpy.org/numpy-tutorials/"},
    #       {"name": "NEPs", "url": "https://numpy.org/neps"}
    #       ],
    "header_links_before_dropdown": 6,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],

}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["source/_static"]

import sys

sys.path.append("../src/")
