# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "SurPyval"
copyright = "2020-" + str(datetime.datetime.now().year) + ", Derryn Knife"
author = "Derryn Knife"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.autodoc']
extensions = [
    "sphinx.ext.napoleon",
    # 'sphinx.ext.doctest',
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "jupyter_sphinx",
]

# Code in ``.. jupyter-execute::`` directives runs in a fresh kernel per
# document at build time; outputs and matplotlib figures are embedded in
# the built HTML, so they always reflect the installed version of surpyval.
jupyter_execute_default_kernel = "python3"

# copybutton_prompt_text = ">>> "
copybutton_prompt_text = (
    r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True

master_doc = "index"

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
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/logo.png"

autosectionlabel_prefix_document = True  # prevent label clashes across pages

# ``autosectionlabel`` mints a cross-reference target from every section
# heading. Prefixing by document stops two *pages* colliding, but not two
# headings within one page -- and the changelog necessarily repeats
# "Serialisation", "Degradation", "Regression" and so on once per
# release. That produced a dozen duplicate-label warnings per build,
# which is most of the noise that hid three genuinely broken autodoc
# targets for who knows how long.
#
# Depth 1 keeps labels for page titles, which is what a ``:ref:`` into
# another page actually wants, and stops minting them for the
# subsections underneath. Nothing referenced those subsection labels
# (the changelog has no inbound ``:ref:`` at all), so nothing breaks.
autosectionlabel_maxdepth = 1

autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
autosummary_generate = True  # Make _autosummary files and include them

html_theme_options = {
    # 'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    # 'analytics_anonymize_ip': False,
    "logo_only": True,
    "style_nav_header_background": "#ea5454",
}


# ``html_theme`` is set above and that is all a modern sphinx_rtd_theme
# needs -- it registers itself as an entry point. The block that used to
# live here re-set the theme and computed ``html_theme_path`` from
# ``sphinx_rtd_theme.get_html_theme_path()``, which the theme has since
# deprecated: it emitted a warning on every build, and its own message
# says the call is safe to remove.
