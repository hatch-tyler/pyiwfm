# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the source directory to the path
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "pyiwfm"
copyright = "2024, California Department of Water Resources"
author = "DWR"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

# -- Plot directive settings -------------------------------------------------
# Configuration for matplotlib plot rendering in documentation
plot_include_source = True
plot_html_show_source_link = True
plot_html_show_formats = False
plot_formats = [("png", 150)]
plot_rcparams = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.constrained_layout.use": True,
}
# Working directory for plot scripts
plot_working_directory = os.path.abspath("../src")

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Templates path
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/CA-DWR/pyiwfm",
    "show_toc_level": 2,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyiwfm/",
            "icon": "fa-brands fa-python",
        },
    ],
    "logo": {
        "text": "pyiwfm",
    },
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

html_context = {
    "github_user": "CA-DWR",
    "github_repo": "pyiwfm",
    "github_version": "main",
    "doc_path": "docs",
}

html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs"],
}

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
