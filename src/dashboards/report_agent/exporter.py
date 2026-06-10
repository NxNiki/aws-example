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
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from dashboards import confluence_client

# The storage-format HTML builders live in export_html (plotly-free) so the
# React dashboard's export endpoint (dashboard_api) can share them. Aliased to
# the original private names to keep this module's call sites unchanged.
from dashboards.report_agent.export_html import REPORT_HEADING_TEXT  # noqa: F401  (re-exported)
from dashboards.report_agent.export_html import (
    build_references_block as _build_references_block,
    build_report_section as _build_report_section,
    figure_block_html as _figure_block_html,
    splice_report_into_storage as _splice_report_into_storage,
)

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    page_id: str
    figures_uploaded: int
    page_version: Optional[int]
    references_listed: int = 0


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
