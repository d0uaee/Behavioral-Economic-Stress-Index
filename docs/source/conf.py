import os
import sys

project   = "BESI — Behavioral Economic Stress Index"
copyright = "2026, Douae Ahadji & Adama Basse — ENSAM Meknes"
author    = "Douae Ahadji & Adama Basse"
release   = "3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

html_theme         = "sphinx_rtd_theme"
html_static_path   = []
templates_path     = ["_templates"]
exclude_patterns   = []
language           = "fr"

html_theme_options = {
    "navigation_depth"    : 4,
    "titles_only"         : False,
    "collapse_navigation" : False,
    "sticky_navigation"   : True,
    "includehidden"       : True,
    "logo_only"           : False,
    "prev_next_buttons_location": "bottom",
}

html_context = {
    "display_github" : True,
    "github_user"    : "d0uaee",
    "github_repo"    : "Behavioral-Economic-Stress-Index",
    "github_version" : "main",
    "conf_py_path"   : "/docs/source/",
}
