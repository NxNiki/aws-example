"""Confluence storage-format HTML builders for the Dashboard Report region.

Pure string functions, deliberately free of plotly/dash imports so BOTH report
exporters can share them: the legacy Dash exporter (``exporter.py``, renders
PNGs server-side via kaleido) and the React dashboard's ``dashboard_api``
export endpoint (receives client-rendered ECharts PNGs).

Idempotency model: everything is written into a heading-bracketed region
(``<h1>Dashboard Report</h1> …``). On re-export the region is replaced in
place; any hand-written content above the heading is preserved.
"""

from __future__ import annotations

import re
from typing import Dict, List

REPORT_HEADING_TEXT = "Dashboard Report"

REPORT_HEADING_RE = re.compile(
    rf"<h1[^>]*>\s*{re.escape(REPORT_HEADING_TEXT)}\s*</h1>",
    re.IGNORECASE,
)


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap_paragraphs(text: str) -> str:
    """Convert plain-text prose to a sequence of ``<p>`` blocks for Confluence storage."""
    if not text or not text.strip():
        return ""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    parts: List[str] = []
    for para in paragraphs:
        clean = escape_html(para.strip()).replace("\n", "<br/>")
        if clean:
            parts.append(f"<p>{clean}</p>")
    return "".join(parts)


def figure_block_html(*, index: int, filename: str, description: str) -> str:
    body = wrap_paragraphs(description) or "<p><em>(no description)</em></p>"
    return (
        f"<h2>Figure {index}</h2>"
        f'<ac:image ac:width="800"><ri:attachment ri:filename="{filename}"/></ac:image>'
        f"{body}"
    )


def build_references_block(references: List[Dict[str, str]]) -> str:
    """``<h2>References</h2>`` followed by a bulleted list of titled links.

    Empty list → empty string (caller skips the section). External URLs
    that didn't fetch are still listed so the reader can follow up.
    """
    if not references:
        return ""
    items: List[str] = []
    for ref in references:
        url = (ref.get("url") or "").strip()
        if not url:
            continue
        title = (ref.get("title") or url).strip()
        items.append(f'<li><a href="{escape_html(url)}">{escape_html(title)}</a></li>')
    if not items:
        return ""
    return "<h2>References</h2><ul>" + "".join(items) + "</ul>"


def build_report_section(
    summary: str,
    figure_blocks: List[str],
    references_block: str = "",
) -> str:
    """Assemble the full bracketed region: heading + summary + per-figure blocks + references."""
    parts: List[str] = [f"<h1>{REPORT_HEADING_TEXT}</h1>"]
    if summary and summary.strip():
        parts.append("<h2>Summary</h2>")
        parts.append(wrap_paragraphs(summary))
    parts.extend(figure_blocks)
    if references_block:
        parts.append(references_block)
    return "".join(parts)


def splice_report_into_storage(existing_storage: str, report_html: str) -> str:
    """
    Replace any existing ``Dashboard Report`` region with ``report_html``.

    If the heading isn't present, append the region. Anything before the
    heading (or before the appended region) stays untouched.
    """
    if not existing_storage:
        return report_html
    match = REPORT_HEADING_RE.search(existing_storage)
    if match:
        return existing_storage[: match.start()] + report_html
    return existing_storage + report_html
