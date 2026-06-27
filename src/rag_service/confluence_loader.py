"""
Resolve ``ConfluenceSource`` entries into raw page content.

Wraps the existing ``bituslabs_ds.confluence.client`` helpers so we don't
duplicate auth or HTML-stripping logic. The loader expands page trees,
deduplicates, and yields ``(page_id, title, text, url, space_key)``.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from rag_service.config import ConfluenceSource

logger = logging.getLogger(__name__)


PageRecord = Tuple[str, str, str, str, str]  # page_id, title, text, url, space_key


def _client():
    """Lazy import so the module loads even when atlassian-python-api is absent."""
    from bituslabs_ds.confluence.client import _get_client

    return _get_client()


def _strip(html: str) -> str:
    from bituslabs_ds.confluence.client import _strip_html

    return _strip_html(html or "")


def _fetch_page(confluence, page_id: str) -> Dict:
    return confluence.get_page_by_id(page_id, expand="body.storage,space")


def _children(confluence, page_id: str) -> List[Dict]:
    out: List[Dict] = []
    start = 0
    while True:
        batch = confluence.get_page_child_by_type(
            page_id, type="page", start=start, limit=50, expand="body.storage,space"
        )
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 50:
            break
        start += 50
    return out


def _pages_in_space(confluence, space_key: str) -> List[Dict]:
    out: List[Dict] = []
    start = 0
    while True:
        batch = confluence.get_all_pages_from_space(space_key, start=start, limit=50, expand="body.storage,space")
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 50:
            break
        start += 50
    return out


def _to_record(page: Dict, source_name: str) -> PageRecord:
    page_id = str(page.get("id", ""))
    title = page.get("title", "Untitled")
    html = (page.get("body") or {}).get("storage", {}).get("value", "")
    text = _strip(html)
    space_key = (page.get("space") or {}).get("key", "")
    url_path = (page.get("_links") or {}).get("webui", "")
    base = (page.get("_links") or {}).get("base", "")
    url = f"{base}{url_path}" if base and url_path else url_path
    return page_id, title, text, url, space_key


def iter_source_pages(source: ConfluenceSource) -> Iterable[PageRecord]:
    """Yield page records for a single source entry, deduplicated by page_id."""
    from bituslabs_ds.confluence.client import list_pages_in_folder

    confluence = _client()
    seen: set = set(source.exclude_page_ids)

    pages: List[Dict] = []

    # Expand folder IDs (Confluence Cloud "folder" content type) to page IDs
    # via the v2 API, then treat them like explicit page_ids.
    folder_page_ids: List[str] = []
    for folder_id in source.folder_ids:
        try:
            folder_page_ids.extend(list_pages_in_folder(folder_id, recurse_subfolders=True))
        except Exception as exc:
            logger.warning("Failed to expand folder %s: %s", folder_id, exc)
    all_page_ids = list(dict.fromkeys(list(source.page_ids) + folder_page_ids))

    # Explicit page IDs (and URL-derived IDs, and folder-expanded IDs)
    for pid in all_page_ids:
        if pid in seen:
            continue
        try:
            pages.append(_fetch_page(confluence, pid))
        except Exception as exc:
            logger.warning("Failed to fetch page_id=%s: %s", pid, exc)

    # Children of any explicit page IDs (one level deep — recurse if needed)
    if source.include_children:
        to_walk = list(all_page_ids)
        while to_walk:
            parent = to_walk.pop()
            try:
                kids = _children(confluence, parent)
            except Exception as exc:
                logger.warning("Failed to list children of %s: %s", parent, exc)
                continue
            for kid in kids:
                kid_id = str(kid.get("id", ""))
                if kid_id and kid_id not in seen:
                    pages.append(kid)
                    to_walk.append(kid_id)

    # Whole-space ingestion
    if source.space_key and not source.page_ids:
        try:
            pages.extend(_pages_in_space(confluence, source.space_key))
        except Exception as exc:
            logger.warning("Failed to crawl space %s: %s", source.space_key, exc)

    for page in pages:
        record = _to_record(page, source.name)
        page_id = record[0]
        if not page_id or page_id in seen or not record[2].strip():
            continue
        seen.add(page_id)
        yield record
