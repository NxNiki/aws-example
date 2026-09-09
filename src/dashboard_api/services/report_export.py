"""Export a composed report to Confluence from client-rendered chart PNGs.

The React-era counterpart of the legacy ``dashboards/report_agent/exporter.py``:
the browser renders each figure with ECharts and ships PNGs
(``chart.getDataURL``), so this service has NO plotting dependency — it decodes
the images, attaches them to the target page, and splices the report region
(``<h1>Dashboard Report</h1> …``, replaced idempotently; content above the
heading is preserved) using the shared ``export_html`` builders and
``confluence_client``.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Any, Optional

from bituslabs_ds.confluence import client as confluence_client
from bituslabs_ds.confluence.export_html import (
    build_references_block,
    build_report_section,
    figure_block_html,
    splice_report_into_storage,
    table_figure_block_html,
)

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Bad page URL / Confluence failure, surfaced to the UI."""


def export_report_pngs(
    *,
    confluence_url: str,
    summary: str,
    figures: list[dict[str, Any]],  # {title, description, png_base64 | html}
    references: list[dict[str, Any]],
) -> tuple[str, int, Optional[int]]:
    """Attach the PNGs and write the report region. Returns
    (page_id, figures_uploaded, new_page_version)."""
    page_id = confluence_client.extract_page_id_from_url(confluence_url)
    if not page_id:
        raise ExportError(f"Could not extract a Confluence page ID from URL: {confluence_url!r}")

    page = confluence_client.get_page_storage(page_id)

    figure_blocks: list[str] = []
    for idx, fig in enumerate(figures, start=1):
        description = fig.get("description") or ""
        title = (fig.get("title") or "").strip()
        if title:
            description = f"{title}\n\n{description}" if description else title

        # Tabular figures (Summary-table grid) ship pre-rendered HTML — embed it
        # inline as a real Confluence table rather than attaching an image.
        table_html = fig.get("html")
        if table_html:
            figure_blocks.append(table_figure_block_html(index=idx, table_html=table_html, description=description))
            continue

        b64 = (fig.get("png_base64") or "").split(",")[-1]  # tolerate a data-URL prefix
        try:
            png_bytes = base64.b64decode(b64, validate=True)
        except Exception as exc:
            raise ExportError(f"Figure {idx} has invalid PNG data: {exc}") from exc
        if not png_bytes:
            logger.warning("Skipping empty figure %d", idx)
            continue
        filename = f"report_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        confluence_client.attach_file(page_id, png_bytes, filename, content_type="image/png")
        figure_blocks.append(figure_block_html(index=idx, filename=filename, description=description))

    references_block = build_references_block(
        [{"url": r.get("url", ""), "title": r.get("title") or ""} for r in references]
    )
    report_html = build_report_section(summary or "", figure_blocks, references_block)
    new_storage = splice_report_into_storage(page["storage"], report_html)

    result = confluence_client.update_page_storage(
        page_id,
        title=page["title"],
        new_storage=new_storage,
        expected_version=page["version"],
    )
    new_version = ((result or {}).get("version") or {}).get("number")
    return page_id, len(figure_blocks), int(new_version) if new_version is not None else None
