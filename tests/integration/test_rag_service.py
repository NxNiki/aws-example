"""
Live RAG service smoke tests.

Hits a running RAG service (local uvicorn or deployed ALB) and asserts
that retrieval still returns relevant, well-scoped passages from the
indexed Confluence corpus. Mirrors the smoke-test questions in
``docs/rag_service.md``.

Service URL resolution order (matches ``rag_service.client``):
    RAG_SERVICE_URL env var > http://localhost:8052

Run against a local service::

    poetry run uvicorn rag_service.app:app --port 8052   # in another terminal
    poetry run pytest tests/integration/test_rag_service.py -v

Against the deployed ALB::

    RAG_SERVICE_URL=http://rag-service-alb-1946335648.us-west-2.elb.amazonaws.com \\
      poetry run pytest tests/integration/test_rag_service.py -v

Tests are skipped if the service is unreachable, so CI stays green when
nothing is running.
"""

from __future__ import annotations

import os
from typing import Tuple

import pytest
import requests

pytestmark = pytest.mark.integration


def _service_url() -> str:
    return os.environ.get("RAG_SERVICE_URL", "http://localhost:8052").rstrip("/")


def _reachable() -> bool:
    try:
        r = requests.get(f"{_service_url()}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


skipif_unreachable = pytest.mark.skipif(
    not _reachable(),
    reason=(
        f"RAG service not reachable at {_service_url()}. "
        "Start it locally with: poetry run uvicorn rag_service.app:app --port 8052"
    ),
)


# (query, expected_source_name, min_top_score)
#
# Source names are checked on the top-1 hit because mixed-source results
# are valid but the most relevant chunk should be in the expected group.
# Score floors are set conservatively below what we observed empirically
# so a chunking/embedding tweak doesn't have to come with a test rewrite.
_QUERIES: list[Tuple[str, str, float]] = [
    ("rollerCoaster math table RTP", "data_slot_games", 0.5),
    ("giftShop free game trigger probability", "data_slot_games", 0.5),
    ("定向奖池分配 RTP 组别", "data_fish_hunter", 0.5),
    ("fish hunter killing intervals streak", "data_fish_hunter", 0.5),
    ("BOOST_POOL DYNAMIC_RTP strategy", "data_fish_hunter", 0.5),
    ("动态规划求解老虎机 RTP", "model_slot_games", 0.5),
]


# ---------------------------------------------------------------------------
# Service-level sanity checks
# ---------------------------------------------------------------------------


@skipif_unreachable
def test_health_reports_loaded_index() -> None:
    resp = requests.get(f"{_service_url()}/health", timeout=5).json()
    assert resp["status"] == "ok"
    assert resp["index_exists"] is True
    assert (resp.get("doc_count") or 0) > 0, "RAG index has zero vectors — rebuild required"


@skipif_unreachable
def test_sources_lists_configured_groups() -> None:
    resp = requests.get(f"{_service_url()}/sources", timeout=5).json()
    names = {s["name"] for s in resp["sources"]}
    # At minimum the three real corpus groups should be present. The placeholder
    # ``model_fish_hunter`` entry is allowed but not required.
    assert {"data_slot_games", "data_fish_hunter", "model_slot_games"}.issubset(names)


# ---------------------------------------------------------------------------
# Retrieval relevance — one parametric test per smoke-test question
# ---------------------------------------------------------------------------


@skipif_unreachable
@pytest.mark.parametrize(("query", "expected_source", "min_score"), _QUERIES)
def test_retrieve_top_hit_matches_expected_source(
    query: str,
    expected_source: str,
    min_score: float,
) -> None:
    resp = requests.post(
        f"{_service_url()}/retrieve",
        json={"query": query, "top_k": 3},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    passages = data["passages"]
    assert passages, f"No passages returned for query: {query!r}"

    top = passages[0]
    assert top["source_name"] == expected_source, (
        f"For query {query!r}: expected top hit from {expected_source!r}, "
        f"got {top['source_name']!r} (title={top['title']!r})"
    )
    assert top["score"] >= min_score, f"For query {query!r}: top score {top['score']:.3f} below floor {min_score}"
