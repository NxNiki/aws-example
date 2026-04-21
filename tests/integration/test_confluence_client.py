"""
Live Confluence API checks (optional).

Set CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_TOKEN (e.g. from ``source .env``)
then run::

    poetry run pytest tests/integration/test_confluence_client.py -v -m integration

Tests are skipped when credentials are missing so CI stays green.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _confluence_configured() -> bool:
    return bool(
        os.environ.get("CONFLUENCE_URL") and os.environ.get("CONFLUENCE_EMAIL") and os.environ.get("CONFLUENCE_TOKEN")
    )


skipif_no_confluence = pytest.mark.skipif(
    not _confluence_configured(),
    reason="Set CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_TOKEN to run Confluence tests",
)


@skipif_no_confluence
def test_list_spaces_returns_list() -> None:
    from dashboards.confluence_client import list_spaces

    spaces = list_spaces()
    assert isinstance(spaces, list)
    if spaces:
        assert "key" in spaces[0] and "name" in spaces[0]


@skipif_no_confluence
def test_search_pages_smoke() -> None:
    from dashboards.confluence_client import search_pages

    pages = search_pages("the", max_results=2)
    assert isinstance(pages, list)
    for p in pages:
        assert "id" in p and "title" in p


@skipif_no_confluence
def test_get_page_content_roundtrip() -> None:
    """Fetch a page by search, then load full body by id."""
    from dashboards.confluence_client import get_page_content, search_pages

    pages = search_pages("a", max_results=1)
    if not pages:
        pytest.skip("No search results to test get_page_content")
    page_id = pages[0]["id"]
    text = get_page_content(page_id)
    assert isinstance(text, str) and len(text) > 0
    assert "# " in text or text.strip()
