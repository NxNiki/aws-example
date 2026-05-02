"""
Confluence client for the AI chat agent.

Provides search and page-reading capabilities so the LLM agent can look up
documentation from Confluence to answer user questions.

Authentication
--------------
Requires three values (from env vars or AWS Secrets Manager via ``_get_secret``):

- ``CONFLUENCE_URL``   — e.g. ``https://yourcompany.atlassian.net``
- ``CONFLUENCE_EMAIL`` — Atlassian account email
- ``CONFLUENCE_TOKEN`` — API token (https://id.atlassian.com/manage-profile/security/api-tokens)

Page cache
----------
Fetched pages are cached in-memory for the process lifetime to avoid
redundant API calls when the agent re-reads the same page.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_page_cache: Dict[str, str] = {}

_PAGE_ID_RE = re.compile(r"/pages/(\d+)")
_TINYURL_RE = re.compile(r"/wiki/x/([A-Za-z0-9_\-]+)")


def _decode_tinyurl_token(token: str) -> Optional[str]:
    """Decode a Confluence tinyurl token (e.g. ``Z4AfOw``) to its numeric page ID.

    Atlassian's encoding is: ``page_id`` → little-endian bytes → URL-safe
    base64, padding stripped. Reversing that is offline and avoids the
    JS-redirect / auth-page traps that defeat HTTP follow-the-redirect.
    """
    import base64

    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded)
        if not raw:
            return None
        return str(int.from_bytes(raw, "little"))
    except Exception as exc:
        logger.debug("Tinyurl token decode failed for %r: %s", token, exc)
        return None


def extract_page_id_from_url(url: str) -> Optional[str]:
    """Pull a numeric page ID out of a Confluence URL.

    Supports both the long form (``.../wiki/spaces/X/pages/12345/Title``)
    and the tinyurl form (``.../wiki/x/<token>``). Tinyurl decoding is
    purely local — no HTTP call required.
    """
    if not url:
        return None
    match = _PAGE_ID_RE.search(url)
    if match:
        return match.group(1)
    tiny_match = _TINYURL_RE.search(url)
    if tiny_match:
        return _decode_tinyurl_token(tiny_match.group(1))
    return None


def extract_page_ids_from_urls(urls: Iterable[str]) -> List[str]:
    """Resolve an iterable of Confluence URLs to deduplicated page IDs.

    Order is preserved (first occurrence wins). URLs that don't yield a
    page ID are silently skipped.
    """
    out: List[str] = []
    seen: set = set()
    for url in urls or []:
        page_id = extract_page_id_from_url(url)
        if page_id and page_id not in seen:
            seen.add(page_id)
            out.append(page_id)
    return out


def _strip_html(html: str) -> str:
    """Convert Confluence storage-format HTML to readable plain text."""
    text = re.sub(r"<br\s*/?>", "\n", html)
    text = re.sub(r"<li[^>]*>", "\n- ", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"<h[1-6][^>]*>", "\n## ", text)
    text = re.sub(r"</h[1-6]>", "\n", text)
    text = re.sub(r"<tr[^>]*>", "\n| ", text)
    text = re.sub(r"<td[^>]*>|<th[^>]*>", " | ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_client():
    """Build an Atlassian Confluence client, lazily imported."""
    try:
        from atlassian import Confluence  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "atlassian-python-api is required for Confluence integration. "
            "Install with:  poetry add atlassian-python-api"
        )

    from dashboards.chat_agent import _get_secret

    url = _get_secret("CONFLUENCE_URL")
    email = _get_secret("CONFLUENCE_EMAIL")
    token = _get_secret("CONFLUENCE_TOKEN")

    if not all([url, email, token]):
        missing = [
            k for k, v in {"CONFLUENCE_URL": url, "CONFLUENCE_EMAIL": email, "CONFLUENCE_TOKEN": token}.items() if not v
        ]
        raise ValueError(
            f"Confluence credentials missing: {', '.join(missing)}. " "Add them to .env or AWS Secrets Manager."
        )

    return Confluence(url=url, username=email, password=token, cloud=True)


def search_pages(query: str, space_key: Optional[str] = None, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search Confluence pages using CQL.

    Returns a list of dicts with keys: id, title, space, url, excerpt.
    """
    confluence = _get_client()

    cql = f'type=page AND text~"{query}"'
    if space_key:
        cql += f' AND space="{space_key}"'

    try:
        raw = confluence.cql(cql, limit=max_results, expand="content.body.view")
        results: Dict[str, Any] = raw if isinstance(raw, dict) else {}
        pages = []
        for item in results.get("results", []):
            content = item.get("content", item)
            excerpt_html = item.get("excerpt", "")
            pages.append(
                {
                    "id": str(content.get("id", "")),
                    "title": content.get("title", "Untitled"),
                    "space": content.get("_expandable", {}).get("space", "").split("/")[-1],
                    "url": content.get("_links", {}).get("webui", ""),
                    "excerpt": _strip_html(excerpt_html)[:300],
                }
            )
        return pages
    except Exception as exc:
        logger.exception("Confluence search failed")
        raise RuntimeError(f"Confluence search failed: {exc}") from exc


def get_page_content(page_id: str) -> str:
    """
    Fetch the full content of a Confluence page by ID.

    Results are cached in-memory for the process lifetime.
    """
    if page_id in _page_cache:
        return _page_cache[page_id]

    confluence = _get_client()

    try:
        raw_page = confluence.get_page_by_id(page_id, expand="body.storage")
        if not isinstance(raw_page, dict):
            raise RuntimeError(f"Unexpected page response type for id={page_id!r}")
        page: Dict[str, Any] = raw_page
        title = str(page.get("title", "Untitled"))
        body = page.get("body")
        if not isinstance(body, dict):
            body = {}
        storage = body.get("storage")
        if not isinstance(storage, dict):
            storage = {}
        html_body = str(storage.get("value", "") or "")
        text = _strip_html(html_body)

        MAX_CHARS = 8000
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + f"\n\n... (truncated, {len(text)} chars total)"

        result = f"# {title}\n\n{text}"
        _page_cache[page_id] = result
        return result
    except Exception as exc:
        logger.exception("Confluence page fetch failed for page_id=%s", page_id)
        raise RuntimeError(f"Failed to fetch Confluence page {page_id}: {exc}") from exc


def list_spaces() -> List[Dict[str, str]]:
    """List available Confluence spaces (key and name)."""
    confluence = _get_client()
    try:
        raw = confluence.get_all_spaces(start=0, limit=50)
        spaces: Dict[str, Any] = raw if isinstance(raw, dict) else {}
        return [{"key": s["key"], "name": s["name"]} for s in spaces.get("results", [])]
    except Exception as exc:
        logger.exception("Confluence list spaces failed")
        raise RuntimeError(f"Failed to list Confluence spaces: {exc}") from exc


def get_page_storage(page_id: str) -> Dict[str, Any]:
    """Return the page's storage-format body, title, and current version number."""
    confluence = _get_client()
    page = confluence.get_page_by_id(page_id, expand="body.storage,version")
    if not isinstance(page, dict):
        raise RuntimeError(f"Unexpected page response type for id={page_id!r}")
    body = (page.get("body") or {}).get("storage") or {}
    version = (page.get("version") or {}).get("number")
    return {
        "id": str(page.get("id", "")),
        "title": str(page.get("title", "Untitled")),
        "storage": str(body.get("value", "") or ""),
        "version": int(version) if version is not None else None,
    }


def attach_file(
    page_id: str,
    file_bytes: bytes,
    filename: str,
    *,
    content_type: str = "image/png",
    comment: str = "",
) -> Dict[str, Any]:
    """Upload an attachment to a page and return its metadata (download URL etc.)."""
    confluence = _get_client()
    import io

    buffer = io.BytesIO(file_bytes)
    buffer.name = filename
    result = confluence.attach_content(
        content=buffer,
        name=filename,
        content_type=content_type,
        page_id=page_id,
        comment=comment or f"Uploaded by report agent: {filename}",
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected attachment response: {result!r}")
    return result


def update_page_storage(
    page_id: str,
    title: str,
    new_storage: str,
    *,
    expected_version: Optional[int] = None,
) -> Dict[str, Any]:
    """Replace a page's storage body. Refreshes the in-process page cache."""
    confluence = _get_client()
    result = confluence.update_page(
        page_id=page_id,
        title=title,
        body=new_storage,
        representation="storage",
        version_comment="Updated by report agent",
        minor_edit=True,
    )
    _page_cache.pop(page_id, None)
    if expected_version is not None:
        new_version = ((result or {}).get("version") or {}).get("number")
        if new_version is not None and new_version <= expected_version:
            logger.warning(
                "Confluence update did not advance version (expected > %s, got %s)",
                expected_version,
                new_version,
            )
    return result if isinstance(result, dict) else {}
