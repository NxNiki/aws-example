"""
User-curated reference loader for the Report tab.

The user adds Confluence URLs (or any URL) via the Report tab's Add
reference button. At LLM-call time and at export time, ``load_references``
resolves each URL to a Confluence page (when possible), fetches the body,
strips it to plain text, and truncates to a per-doc cap so the LLM prompt
stays bounded.

External / unfetchable URLs come back with an empty body and an error
message — the export-side References section still lists them, so the
reader of the Confluence doc can follow up manually.

A small in-process cache keys results by URL so a session that clicks
*Generate description* repeatedly doesn't re-fetch the same pages.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from dashboards import confluence_client

logger = logging.getLogger(__name__)


# Hard cap so a single huge page can't crowd out neighbours in the prompt.
_MAX_CHARS_PER_REFERENCE = 4000

# URL → fetched-entry. Survives across button clicks for the dashboard
# process lifetime; trade-off is staleness if the upstream doc is edited
# during the session, accepted because callers can restart the dashboard.
_REFERENCE_CACHE: Dict[str, Dict[str, str]] = {}


def _is_confluence_link(url: str) -> bool:
    return "/wiki/" in (url or "")


def _fetch_one(url: str, max_chars: int = _MAX_CHARS_PER_REFERENCE) -> Dict[str, str]:
    """Fetch one URL and return ``{url, title, text, error?}``.

    Confluence URLs are resolved through ``extract_page_id_from_url`` and
    fetched via ``get_page_storage``; everything else is recorded as an
    external link with an empty body so the LLM doesn't hallucinate
    contents. Errors are captured so a single bad URL doesn't break the
    whole batch.
    """
    entry: Dict[str, str] = {"url": url, "title": url, "text": ""}
    if not _is_confluence_link(url):
        return entry

    try:
        page_id = confluence_client.extract_page_id_from_url(url)
    except Exception as exc:
        entry["error"] = f"URL parse failed: {exc}"
        return entry
    if not page_id:
        entry["error"] = "URL did not resolve to a Confluence page id"
        return entry

    try:
        page = confluence_client.get_page_storage(page_id)
    except Exception as exc:
        entry["error"] = f"Page fetch failed: {exc}"
        return entry

    title = str(page.get("title") or url)
    storage = str(page.get("storage") or "")
    from dashboards.confluence_client import _strip_html

    text = _strip_html(storage)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated; total {len(text)} chars)"
    entry["title"] = title
    entry["text"] = text
    return entry


def load_references(urls: List[str], *, max_chars: int = _MAX_CHARS_PER_REFERENCE) -> List[Dict[str, str]]:
    """Resolve and fetch every URL, deduped by URL, in input order."""
    seen: set = set()
    out: List[Dict[str, str]] = []
    for url in urls or []:
        cleaned = (url or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        if cleaned not in _REFERENCE_CACHE:
            _REFERENCE_CACHE[cleaned] = _fetch_one(cleaned, max_chars=max_chars)
        out.append(_REFERENCE_CACHE[cleaned])
    return out


def clear_cache() -> None:
    """Drop the in-process cache. Useful between dashboard sessions in tests."""
    _REFERENCE_CACHE.clear()


def format_for_prompt(refs: List[Dict[str, str]]) -> Optional[str]:
    """Concatenate fetched references into the ``# Reference materials`` prompt block.

    Returns ``None`` when the list is empty so the caller can skip the
    section entirely. Each reference is labelled with its title and URL
    so the LLM can cite specifically.
    """
    if not refs:
        return None
    parts: List[str] = []
    for ref in refs:
        title = ref.get("title") or ref.get("url") or "(unnamed)"
        url = ref.get("url") or ""
        body = ref.get("text") or ""
        if not body:
            err = ref.get("error") or "(no body fetched — external link)"
            parts.append(f"--- {title} ({url}) ---\n[{err}]")
        else:
            parts.append(f"--- {title} ({url}) ---\n{body}")
    return "\n\n".join(parts)
