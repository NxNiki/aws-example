"""
Discover and load Confluence reference pages linked from the active doc.

The agent walks every Confluence link in the doc body (References section
included) and recurses one level deeper, capped at ``depth=2``. Page IDs
are deduplicated by ID — not URL string — so different shapes of the same
page are loaded once. URL → page-id parsing is delegated to
``confluence_client.extract_page_ids_from_urls`` (single source of truth).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from dashboards import confluence_client
from dashboards.report_agent.parser import parse

logger = logging.getLogger(__name__)


def _strip_storage_to_text(storage: str) -> str:
    from dashboards.confluence_client import _strip_html

    return _strip_html(storage or "")


def load_references(
    doc_storage: str,
    *,
    depth: int = 2,
    max_pages: int = 30,
    max_chars_per_page: int = 4000,
) -> Dict[str, Dict[str, str]]:
    """Walk Confluence links from ``doc_storage`` up to ``depth`` levels deep.

    Returns ``{page_id: {"title", "text", "url"}}``. Caps total fetched pages
    at ``max_pages`` to bound LLM context cost; per-page text is truncated at
    ``max_chars_per_page`` so a single huge reference can't crowd out neighbours.
    """
    if depth < 1:
        return {}

    parsed = parse(doc_storage)
    seen: Set[str] = set()
    out: Dict[str, Dict[str, str]] = {}
    frontier: List[str] = confluence_client.extract_page_ids_from_urls(parsed.confluence_links)

    for current_depth in range(1, depth + 1):
        next_frontier: List[str] = []
        for page_id in frontier:
            if page_id in seen or len(out) >= max_pages:
                continue
            seen.add(page_id)
            try:
                page = confluence_client.get_page_storage(page_id)
            except Exception as exc:
                logger.warning("Failed to load reference page %s: %s", page_id, exc)
                continue
            text = _strip_storage_to_text(page.get("storage", ""))
            if len(text) > max_chars_per_page:
                text = text[:max_chars_per_page] + f"\n\n... (truncated; {len(text)} chars total)"
            out[page_id] = {
                "title": page.get("title", "Untitled"),
                "text": text,
                "url": f"/wiki/pages/{page_id}",
            }
            if current_depth < depth:
                child_links = parse(page.get("storage", "")).confluence_links
                for child_id in confluence_client.extract_page_ids_from_urls(child_links):
                    if child_id not in seen:
                        next_frontier.append(child_id)
        frontier = next_frontier
        if not frontier or len(out) >= max_pages:
            break

    return out


def format_for_prompt(refs: Dict[str, Dict[str, str]]) -> Optional[str]:
    if not refs:
        return None
    parts = []
    for pid, ref in refs.items():
        parts.append(f"--- Reference: {ref['title']} (page_id={pid}) ---\n{ref['text']}")
    return "\n\n".join(parts)
