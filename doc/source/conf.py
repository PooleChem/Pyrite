# pylint: skip-file

import os
import re
import sys
from pathlib import Path

_ROOT = os.path.abspath("../..")
sys.path.insert(0, _ROOT)
# The user guide's notebooks run in their own kernel, which inherits the environment.
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [_ROOT, os.environ.get("PYTHONPATH")]))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pyrite"
copyright = "2025-2026, M.J. van der Lugt"
author = "M.J. van der Lugt"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "myst_nb",
    "sphinx.ext.viewcode",  # [source] links to the highlighted source code
    "sphinx_copybutton",  # a copy button on code blocks
    "sphinx_design",  # cards, grids, tabs, badges
    "sphinx_togglebutton",  # collapsible blocks
    "sphinxcontrib.mermaid",  # diagrams written as text
    "sphinx_iconify",  # inline icons
    "sphinx_sitemap",  # sitemap.xml for search engines
]
# numpydoc_show_class_members = False

# numpydoc renders the docstrings (napoleon is not used: with both, napoleon converted the sections
# first and numpydoc hardly acted). Link the types in Parameters / Returns:
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "Mol": "pyrite.Mol",
    "Pose": "pyrite.Pose",
    "Poses": "pyrite.Poses",
    "PoseLayout": "pyrite.PoseLayout",
    "AtomType": "pyrite.AtomType",
    "Viewer": "pyrite.Viewer",
    "ScoringFunction": "pyrite.scoring.ScoringFunction",
    "Clamp": "pyrite.scoring.Clamp",
    "GridScore": "pyrite.scoring.grid.GridScore",
    "Dependency": "pyrite.scoring.dependencies.Dependency",
    "KNNDependency": "pyrite.scoring.dependencies.KNNDependency",
    "Realization": "pyrite.scoring.dependencies.Realization",
    "Bounds": "pyrite.bounds.Bounds",
    "Pocket": "pyrite.bounds.Pocket",
    "KDTree": "scipy.spatial.KDTree",
    "OptimizeResult": "scipy.optimize.OptimizeResult",
    "ndarray": "numpy.ndarray",
    "NDArray": "numpy.ndarray",
    "array_like": ":term:`numpy:array_like`",
    "ArrayLike": ":term:`numpy:array_like`",
}
numpydoc_xref_ignore = {
    "optional", "default", "or", "of", "shape", "same", "type", "any", "None", "inf", "-inf",
    "ipywidgets.HTML",  # ipywidgets publishes no inventory for intersphinx
}

# numpydoc validates every documented docstring during the build (warnings in the build output),
# including the extended summary (ES01), See Also (SA01) and Examples (EX01) sections. Left out
# only: the layout of the quotes and blank lines (GL01-GL03).
numpydoc_validation_checks = {"all", "GL01", "GL02", "GL03"}
# Not validated:
# - overrides on subclasses of the hooks and base methods: numpydoc reads an override's own docstring,
#   but the docs show the inherited one (autodoc_inherit_docstrings). The base definitions
#   (ScoringFunction._score, Bounds.is_within, Dependency.compute, ...) are still validated;
# - the attributes listed in a class's Attributes section, which get a page of their own;
# - what AtomType inherits from int, and vina_atom_consts (a dict, which has no docstring).
_OVERRIDES = (
    r"_score|_batch_scores|_score_and_gradient|_score_field|_kernel|_mask|get_dependencies"
    r"|is_within|squared_distance|distance|transform_sample_to_bounds|compute|group_key|merge_group|narrow|row"
    r"|__call__"
)
_BASES = r"scoring\.ScoringFunction|scoring\.dependencies\.Dependency|scoring\.Dependency|bounds\.Bounds|scoring\.protein\._KNNScoringFunction"
numpydoc_validation_exclude = {
    rf"^pyrite\.(?!(?:{_BASES})\.)[\w.]+\.(?:{_OVERRIDES})$",
    r"^pyrite\.scoring\.protein\._KNNScoringFunction\.(?:_score|_batch_scores|get_dependencies)$",
    r"^pyrite\.Poses?\.(?:layout|rotation|translation|torsions)$",
    r"^pyrite\.AtomType\.(?:denominator|imag|numerator|real|from_bytes|to_bytes|as_integer_ratio|bit_count|bit_length|conjugate|is_integer)$",
    r"^pyrite\.vina_atom_consts$",
}
# AtomType is an IntEnum: numpydoc reads the parameters of Enum's functional API as its own.
numpydoc_validation_overrides = {"PR01": [r"Enumeration describing the various atom types"]}


autosummary_generate = True  # <- crucial

# `Name` in single backticks links to Name when it is documented (as numpy's docs do), and is
# italic otherwise (parameter names).
default_role = "autolink"

templates_path = ["_templates"]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
html_title = "Pyrite"
# The public URL: used for the sitemap and the AI links ("Copy page", "Open in ChatGPT/Claude"),
# which point the AI to this page's source in _sources/.
html_baseurl = "https://poolechem.github.io/Pyrite/"
sitemap_url_scheme = "{link}"
html_favicon = "_static/favicon.png"
html_css_files = ["custom.css"]
html_theme_options = {
    "light_logo": "_static/pyrite_logo.png",
    "dark_logo": "_static/pyrite_logo_dark.png",
    "accent_color": "tomato",  # the orange-red of the logo
    "github_url": "https://github.com/PooleChem/Pyrite",
}
# Algolia DocSearch replaces the search box when its keys are set; without them the built-in search is
# used. Apply at https://docsearch.algolia.com/.
if os.getenv("DOCSEARCH_APP_ID"):
    extensions.append("sphinx_docsearch")
    docsearch_app_id = os.environ["DOCSEARCH_APP_ID"]
    docsearch_api_key = os.environ["DOCSEARCH_API_KEY"]
    docsearch_index_name = os.environ["DOCSEARCH_INDEX_NAME"]

# The right-hand sidebar: "On this page" only (no repository stats, edit link or ads). custom.css
# hides it where it is empty.
html_sidebars = {"**": ["sidebars/localtoc.html"]}

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

# py3Dmol outputs "application/3dmoljs_load.v0" next to "text/html": skip the first, so the HTML
# version (an interactive 3D view) is shown in the pages.
# A page of the user guide that fails to run fails the build (instead of a warning and a page
# without its outputs).
nb_execution_raise_on_error = True
# Show the error of a failing cell in the build log, and give a cell up to 10 minutes: the default 30
# seconds is too short for building a grid on a small machine, such as that of the docs workflow.
nb_execution_show_tb = True
nb_execution_timeout = 600
nb_mime_priority_overrides = [("html", "application/3dmoljs_load.v0", None)]
# myst-nb still warns that it skips the 3Dmol type, although the HTML version is shown.
suppress_warnings = ["mystnb.unknown_mime_type"]

# Anchors for the headings of Markdown pages, so [text](#a-heading) links work.
myst_heading_anchors = 3
# ::: fences for sphinx_design directives in Markdown
myst_enable_extensions = ["colon_fence"]

# Copy only the code of ">>> " examples, without the prompts and the output.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


def _hide_ai_links_on_api_pages(app, pagename, templatename, context, doctree):
    """Hide the AI links on pages written by autodoc.

    The AI links send the page's source, which for an API page is only the autodoc directive, not
    the docstrings.
    """
    path = app.env.doc2path(pagename, base=True) if pagename in app.env.all_docs else None
    if path is None or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        if ".. auto" in f.read():
            context["meta"] = {**(context.get("meta") or {}), "hide_ai_links": "true"}


_SIDEBAR_MODULE = re.compile(r'\((<code [^>]*><span class="pre">)([^<]*)(</span></code>)\)')


def _wrap_sidebar_module_names(app, exception):
    """Keep "(pyrite.scoring.dependencies)" together in the left sidebar, breaking it after a dot.

    The name and its parentheses become one ``.module-name`` (see custom.css), which moves to a line of
    its own; ``<wbr>`` after every dot lets a name that is still too long break there. Done on the
    written pages, as a script in the browser would make the sidebar flash on every load.
    """
    if exception is not None or app.builder.format != "html":
        return

    def wrap(match):
        name = match.group(2).replace(".", ".<wbr>")
        return f'<span class="module-name">({match.group(1)}{name}{match.group(3)})</span>'

    for path in Path(app.outdir).rglob("*.html"):
        html = path.read_text(encoding="utf-8")
        start = html.find('<div class="globaltoc"')
        if start < 0:
            continue
        end = html.find("</aside>", start)
        sidebar = _SIDEBAR_MODULE.sub(wrap, html[start:end])
        path.write_text(html[:start] + sidebar + html[end:], encoding="utf-8")


def setup(app):
    app.connect("build-finished", _wrap_sidebar_module_names)
    app.connect("html-page-context", _hide_ai_links_on_api_pages)
