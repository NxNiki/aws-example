"""
Append a Plotly figure to a Confluence page as an image attachment plus a
storage-format ``FIGURE`` block (see ``docs/report_agent.md``).

Used by the dashboard's *Add to doc* button. The block format is the input
contract the report agent reads — keep it stable.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, Optional

import plotly.graph_objects as go

from dashboards import confluence_client

logger = logging.getLogger(__name__)


def _figure_to_png(figure: go.Figure, *, width: int, height: int, scale: float = 2.0) -> bytes:
    """Render a Plotly figure to PNG bytes via kaleido."""
    return figure.to_image(format="png", width=width, height=height, scale=scale)


def _build_figure_storage_block(
    *,
    figure_id: str,
    filename: str,
    caption: str,
    filter_context: Dict[str, Any],
    index: int,
) -> str:
    """Storage-format snippet for one FIGURE block. Mirrors docs/report_agent.md.

    The visible ``<strong>Figure N</strong>`` line is the user-facing handle
    for referencing the figure in ``/prompt:`` lines and reordering by edit.
    """
    safe_caption = (caption or "").strip()
    context_json = json.dumps(filter_context, sort_keys=True, default=str)
    safe_caption_html = safe_caption.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_context_html = context_json.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f"<!-- FIGURE:START id={figure_id} -->"
        f"<p><strong>Figure {index}</strong></p>"
        '<ac:image ac:width="800">'
        f'<ri:attachment ri:filename="{filename}"/>'
        "</ac:image>"
        '<ac:structured-macro ac:name="info">'
        "<ac:rich-text-body>"
        f"<p><strong>Caption:</strong> {safe_caption_html}</p>"
        f"<p><strong>Context:</strong> <code>{safe_context_html}</code></p>"
        "</ac:rich-text-body>"
        "</ac:structured-macro>"
        f"<!-- FIGURE:END id={figure_id} -->"
    )


_REFERENCES_RE = re.compile(
    r"<h[1-6][^>]*>\s*References\s*</h[1-6]>",
    re.IGNORECASE,
)
_FIGURE_START_COUNT_RE = re.compile(r"<!--\s*FIGURE:START\s+id=", re.IGNORECASE)


def _count_existing_figures(storage: str) -> int:
    return len(_FIGURE_START_COUNT_RE.findall(storage or ""))


def _splice_block_into_storage(existing_storage: str, new_block: str) -> str:
    """
    Insert ``new_block`` into the page body just before any References heading
    (per the doc grammar). If no References section exists yet, append to the
    end. Existing storage may be empty for a fresh page.
    """
    if not existing_storage:
        return new_block
    match = _REFERENCES_RE.search(existing_storage)
    if match:
        return existing_storage[: match.start()] + new_block + existing_storage[match.start() :]
    return existing_storage + new_block


def add_figure_to_page(
    page_url: str,
    figure: go.Figure,
    *,
    caption: str = "",
    filter_context: Optional[Dict[str, Any]] = None,
    width: int = 1200,
    height: int = 600,
) -> Dict[str, Any]:
    """
    Render the figure to PNG, upload as a Confluence attachment, and append a
    FIGURE block referencing it to the page identified by ``page_url``.

    Returns a small status dict ``{"figure_id", "filename", "page_version"}``
    on success. Raises with a descriptive message on failure.
    """
    page_id = confluence_client.extract_page_id_from_url(page_url)
    if not page_id:
        raise ValueError(f"Could not extract a Confluence page ID from URL: {page_url!r}")

    figure_id = uuid.uuid4().hex
    filename = f"figure_{int(time.time())}_{figure_id[:8]}.png"

    png_bytes = _figure_to_png(figure, width=width, height=height)
    confluence_client.attach_file(page_id, png_bytes, filename, content_type="image/png")

    page = confluence_client.get_page_storage(page_id)
    next_index = _count_existing_figures(page["storage"]) + 1
    new_block = _build_figure_storage_block(
        figure_id=figure_id,
        filename=filename,
        caption=caption,
        filter_context=filter_context or {},
        index=next_index,
    )
    new_storage = _splice_block_into_storage(page["storage"], new_block)

    confluence_client.update_page_storage(
        page_id,
        title=page["title"],
        new_storage=new_storage,
        expected_version=page["version"],
    )
    logger.info("Added figure %s to page %s as %s", figure_id, page_id, filename)
    return {
        "figure_id": figure_id,
        "filename": filename,
        "page_id": page_id,
    }
