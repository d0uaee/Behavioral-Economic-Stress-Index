from __future__ import annotations

from datetime import datetime

project = "BESI V3"
author = "Douae Ahadji & Adama Basse"
copyright = f"{datetime.now().year}, {author}"
release = "V3"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".md": "markdown",
}

master_doc = "index"

language = "fr"

html_theme = "furo"
html_title = "BESI V3"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

autosectionlabel_prefix_document = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
]

