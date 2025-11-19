# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.insert(0, os.path.abspath("../../../api/src"))  # nv-ingest-api src
sys.path.insert(1, os.path.abspath("../../../client/src"))  # nv-ingest-client src
sys.path.insert(2, os.path.abspath("../../../src"))  # nv-ingest src

project = "nv-ingest"
copyright = "2025, Nvidia"
author = "Nvidia"
release = "24.12"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "swagger_plugin_for_sphinx",
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"

html_theme_options = {
    "header_links": [
        ("Home", "index"),
        ("GitHub", "https://github.com/NVIDIA/nvidia-sphinx-theme", True, "fab fa-github"),
    ],
    "footer_links": [
        ("Privacy Policy", "https://www.nvidia.com/en-us/about-nvidia/privacy-policy/"),
        ("Terms of Use", "https://www.nvidia.com/en-us/about-nvidia/legal-info/"),
    ],
    "show_prev_next": True,  # Show next/previous buttons at bottom
}

html_static_path = ["_static"]
