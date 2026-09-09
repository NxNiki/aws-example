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
_FOLDER_ID_RE = re.compile(r"/folder/(\d+)")


def _decode_tinyurl_token(token: str) -> Optional[str]:
    """Decode a legacy Confluence tinyurl token (e.g. ``Z4AfOw``) to its numeric page ID.

    Works for the older base64-of-little-endian-page-id encoding. Newer
    Confluence Cloud short codes (random-looking 4-5 char tokens) are NOT
    decodable offline — those go through ``_resolve_tinyurl_via_http``.
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


def _resolve_tinyurl_via_http(url: str) -> Optional[str]:
    """Authenticated HTTP fallback for tokens the offline decoder can't crack.

    Confluence Cloud responds either with a 302 to ``.../pages/<id>/...``
    or with a 200 HTML page that embeds the page ID in meta tags / JS
    variables. We check the redirect chain first, then scrape the body
    as a last resort.
    """
    try:
        import requests

        from bituslabs_ds.aws_secrets import get_secret as _get_secret
    except ImportError:
        return None

    email = _get_secret("CONFLUENCE_EMAIL")
    token = _get_secret("CONFLUENCE_TOKEN")
    auth = (email, token) if (email and token) else None
    try:
        resp = requests.get(
            url,
            auth=auth,
            allow_redirects=True,
            timeout=10,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
    except Exception as exc:
        logger.warning("Tinyurl HTTP resolve failed for %s: %s", url, exc)
        return None

    candidates = [resp.url] + [h.url for h in (resp.history or [])]
    for candidate in candidates:
        if candidate:
            match = _PAGE_ID_RE.search(candidate)
            if match:
                return match.group(1)

    body = resp.text or ""
    meta_match = re.search(r'name="ajs-page-id"[^>]*content="(\d+)"', body, re.IGNORECASE)
    if meta_match:
        return meta_match.group(1)
    pid_match = re.search(r'pageId["\s:=]+(\d{6,})', body)
    if pid_match:
        return pid_match.group(1)
    return None


def extract_page_id_from_url(url: str) -> Optional[str]:
    """Pull a numeric page ID out of a Confluence URL.

    Resolution order:
      1. Long-form ``.../wiki/spaces/X/pages/12345/Title`` (regex match).
      2. Legacy tinyurl ``.../wiki/x/<token>`` decoded offline.
      3. New-style tinyurl resolved via authenticated HTTP redirect chain.
    """
    if not url:
        return None
    match = _PAGE_ID_RE.search(url)
    if match:
        return match.group(1)
    tiny_match = _TINYURL_RE.search(url)
    if tiny_match:
        offline = _decode_tinyurl_token(tiny_match.group(1))
        if offline:
            return offline
        return _resolve_tinyurl_via_http(url)
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


def extract_folder_id_from_url(url: str) -> Optional[str]:
    """Pull a numeric folder ID out of a Confluence ``/folder/<id>`` URL."""
    if not url:
        return None
    match = _FOLDER_ID_RE.search(url)
    return match.group(1) if match else None


def extract_folder_ids_from_urls(urls: Iterable[str]) -> List[str]:
    """Resolve an iterable of Confluence URLs to deduplicated folder IDs.

    URLs that aren't ``/folder/<id>`` shaped (e.g. plain page URLs) are
    silently skipped.
    """
    out: List[str] = []
    seen: set = set()
    for url in urls or []:
        folder_id = extract_folder_id_from_url(url)
        if folder_id and folder_id not in seen:
            seen.add(folder_id)
            out.append(folder_id)
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

    from bituslabs_ds.aws_secrets import get_secret as _get_secret

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


def _v2_session():
    """Return ``(requests.Session, base_url)`` for Confluence Cloud REST v2 calls.

    The v1 client wrapped by ``atlassian-python-api`` doesn't expose folder
    endpoints, so v2 is reached directly via authenticated HTTP.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError("The 'requests' package is required for Confluence v2 API calls.") from exc

    from bituslabs_ds.aws_secrets import get_secret as _get_secret

    url = _get_secret("CONFLUENCE_URL")
    email = _get_secret("CONFLUENCE_EMAIL")
    token = _get_secret("CONFLUENCE_TOKEN")
    if not all([url, email, token]):
        missing = [
            k for k, v in {"CONFLUENCE_URL": url, "CONFLUENCE_EMAIL": email, "CONFLUENCE_TOKEN": token}.items() if not v
        ]
        raise ValueError(f"Confluence credentials missing: {', '.join(missing)}.")

    session = requests.Session()
    session.auth = (str(email), str(token))
    session.headers.update({"Accept": "application/json"})
    base = str(url).rstrip("/") + "/wiki/api/v2"
    return session, base


def list_pages_in_folder(folder_id: str, *, recurse_subfolders: bool = True) -> List[str]:
    """Return page IDs that live (transitively) under a Confluence folder.

    Uses the Cloud REST v2 ``folders/{id}/direct-children`` endpoint and
    walks any nested subfolders when ``recurse_subfolders`` is true. Other
    content types (whiteboards, blog posts, databases) are ignored.
    """
    import urllib.parse

    session, base = _v2_session()
    seen_folders: set = set()
    seen_pages: set = set()
    page_ids: List[str] = []
    stack: List[str] = [str(folder_id)]

    while stack:
        fid = stack.pop()
        if fid in seen_folders:
            continue
        seen_folders.add(fid)

        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = session.get(f"{base}/folders/{fid}/direct-children", params=params, timeout=15)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("Failed to list children of folder %s: %s", fid, exc)
                break

            data = resp.json()
            if not isinstance(data, dict):
                break
            for child in data.get("results") or []:
                child_id = str(child.get("id") or "")
                child_type = child.get("type")
                if not child_id:
                    continue
                if child_type == "page":
                    if child_id not in seen_pages:
                        seen_pages.add(child_id)
                        page_ids.append(child_id)
                elif child_type == "folder" and recurse_subfolders:
                    stack.append(child_id)

            next_link = (data.get("_links") or {}).get("next")
            if not next_link:
                break
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(next_link).query)
            cursor = (qs.get("cursor") or [None])[0]
            if not cursor:
                break

    return page_ids


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
    """Upload (create or update-in-place) a page attachment; returns its metadata.

    Raw multipart requests, NOT ``Confluence.attach_content``: that helper
    re-posts a consumed stream when the name already exists, silently
    replacing the attachment with a 0-byte version (figures then render as
    blank/invisible on the page).
    """
    session, v2_base = _v2_session()
    v1_base = v2_base.replace("/api/v2", "/rest/api")
    headers = {"X-Atlassian-Token": "nocheck"}
    data = {"minorEdit": "true", "comment": comment or f"Uploaded by report agent: {filename}"}

    resp = session.get(f"{v1_base}/content/{page_id}/child/attachment", params={"filename": filename}, timeout=30)
    resp.raise_for_status()
    existing = resp.json().get("results", [])
    url = (
        f"{v1_base}/content/{page_id}/child/attachment/{existing[0]['id']}/data"
        if existing
        else f"{v1_base}/content/{page_id}/child/attachment"
    )
    resp = session.post(
        url, headers=headers, data=data, files={"file": (filename, file_bytes, content_type)}, timeout=60
    )
    resp.raise_for_status()
    result = resp.json()
    result = result["results"][0] if "results" in result else result
    size = (result.get("extensions") or {}).get("fileSize")
    if size is not None and size != len(file_bytes):
        raise RuntimeError(f"attachment {filename} uploaded {size} bytes, expected {len(file_bytes)}")
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
