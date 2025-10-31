# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyxalign"
copyright = "2025, Argonne National Laboratory"
author = "Hanna Ruth"
release = "0.1.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

# MyST-NB configuration
nb_execution_mode = "off"  # Don't re-execute notebooks
nb_render_image_options = {"align": "center"}

# Enable processing of notebook attachments (pasted images)
nb_output_stderr = "show"
nb_merge_streams = True

# Configure MIME type priorities for different output formats
nb_mime_priority_overrides = [
    ('html', 'image/png', 10),
    ('html', 'image/jpeg', 10),
    ('html', 'image/svg+xml', 10),
    ('html', 'text/html', 20),
]

# Ensure MyST parser is configured
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# Configure autodoc to respect __all__
autodoc_default_options = {
    "members": True,  # Document members
    "undoc-members": False,  # Don't document undocumented members
    "show-inheritance": True,
    "imported-members": False,  # Don't document imported members
    "member-order": "bysource",
}
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

