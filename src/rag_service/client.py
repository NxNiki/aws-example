"""
HTTP client used by the chat agent (and any other caller) to talk to the
RAG microservice.

Keeps the agent decoupled from OpenSearch / embedding details — the agent
only knows the service's URL.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


def _service_url() -> str:
    return os.environ.get("RAG_SERVICE_URL", "http://localhost:8052").rstrip("/")


def retrieve_passages(
    query: str,
    top_k: int = 5,
    timeout: float = 10.0,
    base_url: Optional[str] = None,
) -> str:
    """Call the RAG service /retrieve endpoint and return a formatted string.

    Returns a human-readable block suitable for an LLM tool response. On any
    error returns an explanatory message rather than raising — the agent
    should keep working even if the service is down.
    """
    url = (base_url or _service_url()) + "/retrieve"
    try:
        resp = requests.post(
            url,
            json={"query": query, "top_k": top_k},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("RAG service request failed: %s", exc)
        return f"RAG service unavailable ({exc}). " "Falling back: try `search_confluence` for live keyword search."

    data = resp.json()
    passages: List[dict] = data.get("passages", [])
    if not passages:
        return f"No Confluence passages found for: {query}"

    lines = [f"Top {len(passages)} Confluence passages for '{query}':\n"]
    for p in passages:
        header = f"### {p['title']} (page_id={p['page_id']}, score={p['score']:.3f})"
        if p.get("url"):
            header += f" — {p['url']}"
        lines.append(header)
        lines.append(p.get("text", ""))
        lines.append("")
    return "\n".join(lines)


def health(base_url: Optional[str] = None, timeout: float = 5.0) -> dict:
    url = (base_url or _service_url()) + "/health"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
