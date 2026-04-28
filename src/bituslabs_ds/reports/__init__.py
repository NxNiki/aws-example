"""Shared HTML report scaffolding: base template, styles, and JS helpers.

See :func:`render_page` for the primary entry point. Assets in
``src/bituslabs_ds/reports/assets/`` are inlined into the output HTML, so the
resulting page is self-contained (no external file dependencies at serve time).
"""

from bituslabs_ds.reports.render import (
    escape_json_for_html_script,
    load_asset,
    load_template,
    render_page,
)

__all__ = [
    "escape_json_for_html_script",
    "load_asset",
    "load_template",
    "render_page",
]
