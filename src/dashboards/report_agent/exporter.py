"""
Export the Report tab's content (overall summary + figures with descriptions)
to a Confluence page.

Idempotency model
-----------------
The exporter writes everything into a heading-bracketed region:
``<h1>Dashboard Report</h1> ... (figures + descriptions) ...``

On subsequent exports, the existing region is replaced in place. Any content
*above* the ``Dashboard Report`` heading is preserved, so users can keep
hand-written notes at the top of the doc.

This is the only direction of interaction between the dashboard and
Confluence — the dashboard pushes a snapshot, and nothing reads back from
the doc.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from dashboards import confluence_client

logger = logging.getLogger(__name__)


REPORT_HEADING_TEXT = "Dashboard Report"

_REPORT_HEADING_RE = re.compile(
    rf"<h1[^>]*>\s*{re.escape(REPORT_HEADING_TEXT)}\s*</h1>",
    re.IGNORECASE,
)


@dataclass
class ExportResult:
    page_id: str
    figures_uploaded: int
    page_version: Optional[int]
    references_listed: int = 0


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _wrap_paragraphs(text: str) -> str:
    """Convert plain-text prose to a sequence of ``<p>`` blocks for Confluence storage."""
    if not text or not text.strip():
        return ""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    parts: List[str] = []
    for para in paragraphs:
        clean = _escape_html(para.strip()).replace("\n", "<br/>")
        if clean:
            parts.append(f"<p>{clean}</p>")
    return "".join(parts)


def _figure_block_html(*, index: int, filename: str, description: str) -> str:
    body = _wrap_paragraphs(description) or "<p><em>(no description)</em></p>"
    return (
        f"<h2>Figure {index}</h2>"
        f'<ac:image ac:width="800"><ri:attachment ri:filename="{filename}"/></ac:image>'
        f"{body}"
    )


def _build_references_block(references: List[Dict[str, str]]) -> str:
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
        items.append(f'<li><a href="{_escape_html(url)}">{_escape_html(title)}</a></li>')
    if not items:
        return ""
    return "<h2>References</h2><ul>" + "".join(items) + "</ul>"


def _build_report_section(
    summary: str,
    figure_blocks: List[str],
    references_block: str = "",
) -> str:
    """Assemble the full bracketed region: heading + summary + per-figure blocks + references."""
    parts: List[str] = [f"<h1>{REPORT_HEADING_TEXT}</h1>"]
    if summary and summary.strip():
        parts.append("<h2>Summary</h2>")
        parts.append(_wrap_paragraphs(summary))
    parts.extend(figure_blocks)
    if references_block:
        parts.append(references_block)
    return "".join(parts)


def _splice_report_into_storage(existing_storage: str, report_html: str) -> str:
    """
    Replace any existing ``Dashboard Report`` region with ``report_html``.

    If the heading isn't present, append the region. Anything before the
    heading (or before the appended region) stays untouched.
    """
    if not existing_storage:
        return report_html
    match = _REPORT_HEADING_RE.search(existing_storage)
    if match:
        return existing_storage[: match.start()] + report_html
    return existing_storage + report_html


def _figure_to_png(figure: go.Figure, *, width: int, height: int, scale: float = 2.0) -> bytes:
    return figure.to_image(format="png", width=width, height=height, scale=scale)


def export_report(
    *,
    page_url: str,
    summary: str,
    figures: List[Dict[str, Any]],
    references: Optional[List[Dict[str, str]]] = None,
) -> ExportResult:
    """Render figures, attach to the page, and write the report region.

    ``figures`` items are the entries from ``report-figures`` store. Each
    must have:
      - ``fig_dict``: full Plotly figure dict (data + layout)
      - ``description``: prose to render below the image
      - ``export_width`` / ``export_height``: PNG dimensions

    ``references`` items are the output of
    ``report_agent.references.load_references``: ``{url, title, text?}``.
    They render as a trailing ``<h2>References</h2>`` block; deduped by URL.
    """
    page_id = confluence_client.extract_page_id_from_url(page_url)
    if not page_id:
        raise ValueError(f"Could not extract a Confluence page ID from URL: {page_url!r}")

    page = confluence_client.get_page_storage(page_id)

    figure_blocks: List[str] = []
    for idx, fig in enumerate(figures, start=1):
        fig_dict = fig.get("fig_dict") or {}
        if not fig_dict.get("data"):
            logger.warning("Skipping empty figure at index %d", idx)
            continue
        width = int(fig.get("export_width") or 1200)
        height = int(fig.get("export_height") or 600)
        png_bytes = _figure_to_png(go.Figure(fig_dict), width=width, height=height)
        filename = f"report_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        confluence_client.attach_file(page_id, png_bytes, filename, content_type="image/png")
        figure_blocks.append(
            _figure_block_html(
                index=idx,
                filename=filename,
                description=fig.get("description") or "",
            )
        )

    references_block = _build_references_block(references or [])
    report_html = _build_report_section(summary or "", figure_blocks, references_block)
    new_storage = _splice_report_into_storage(page["storage"], report_html)

    result = confluence_client.update_page_storage(
        page_id,
        title=page["title"],
        new_storage=new_storage,
        expected_version=page["version"],
    )
    new_version = ((result or {}).get("version") or {}).get("number")
    return ExportResult(
        page_id=page_id,
        figures_uploaded=len(figure_blocks),
        page_version=int(new_version) if new_version is not None else None,
        references_listed=len([r for r in (references or []) if r.get("url")]),
    )
