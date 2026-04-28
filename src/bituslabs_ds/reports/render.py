"""Self-contained HTML report rendering with shared CSS + JS helpers.

The goal of this module is to keep Python data-analysis scripts free of
HTML/CSS/JS boilerplate. Per-script plot code can be written in a real
``.js`` file (editor highlighting, no f-string brace doubling) and passed
to :func:`render_page` as ``inline_scripts``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_ASSETS_DIR = Path(__file__).parent / "assets"
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"


def escape_json_for_html_script(s: str) -> str:
    """Escape ``</`` inside a JSON blob so it can be inlined in a ``<script>`` tag."""
    return s.replace("</", "<\\/")


@lru_cache(maxsize=None)
def load_asset(name: str) -> str:
    """Read a bundled asset (``base.html`` / ``base.css`` / ``plotly_utils.js``)."""
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


def load_template(path: str | Path) -> str:
    """Read an arbitrary template file (e.g., a per-script ``.js`` file)."""
    return Path(path).read_text(encoding="utf-8")


def render_page(
    title: str,
    body: str,
    *,
    extra_styles: str = "",
    extra_head: str = "",
    inline_scripts: str = "",
    include_plotly: bool = True,
    include_plotly_utils: bool = True,
) -> str:
    """Render a self-contained HTML page with shared base styles and helper JS.

    Parameters
    ----------
    title
        ``<title>`` text (caller escapes if untrusted).
    body
        HTML inserted inside ``<body>`` before any trailing scripts.
    extra_styles
        Additional CSS appended after ``base.css``.
    extra_head
        Raw HTML inserted inside ``<head>`` (extra ``<meta>`` / ``<link>`` / ``<script>``).
    inline_scripts
        Raw JS inlined in a ``<script>`` at end of ``<body>``. When
        ``include_plotly_utils`` is True, ``window.ReportUtils`` helpers are
        available to this script.
    include_plotly
        Emit the Plotly.js CDN ``<script>`` tag in ``<head>``.
    include_plotly_utils
        Inline ``plotly_utils.js`` (exposes ``window.ReportUtils``) before caller scripts.
    """
    template = load_asset("base.html")
    base_css = load_asset("base.css")

    head_parts: list[str] = []
    if include_plotly:
        head_parts.append(f'<script src="{_PLOTLY_CDN}"></script>')
    if extra_head:
        head_parts.append(extra_head)
    head_html = "\n".join(head_parts)

    styles = base_css if not extra_styles else f"{base_css}\n{extra_styles}"

    script_parts: list[str] = []
    if include_plotly_utils:
        script_parts.append(f"<script>\n{load_asset('plotly_utils.js')}\n</script>")
    if inline_scripts:
        script_parts.append(f"<script>\n{inline_scripts}\n</script>")
    scripts_html = "\n".join(script_parts)

    return (
        template.replace("{{TITLE}}", title)
        .replace("{{HEAD}}", head_html)
        .replace("{{STYLES}}", styles)
        .replace("{{BODY}}", body)
        .replace("{{SCRIPTS}}", scripts_html)
    )
