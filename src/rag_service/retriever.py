"""
Query-time retrieval.

Performs BM25 + vector kNN searches against the same index and merges with
Reciprocal Rank Fusion (RRF). RRF works on every OpenSearch flavor including
Serverless, where the ``hybrid`` query plugin may not be available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from rag_service.config import RagSettings, load_settings
from rag_service.embeddings import Embedder
from rag_service.opensearch_client import get_client

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    page_id: str
    title: str
    text: str
    url: str
    space_key: str
    source_name: str
    chunk_index: int
    score: float


_RRF_K = 60  # standard RRF dampening constant


def _bm25_search(client, index: str, query: str, k: int) -> List[Dict]:
    body = {
        "size": k,
        "query": {"match": {"text": query}},
    }
    return client.search(index=index, body=body)["hits"]["hits"]


def _knn_search(client, index: str, vector: List[float], k: int) -> List[Dict]:
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
    }
    return client.search(index=index, body=body)["hits"]["hits"]


def _rrf_merge(rankings: List[List[Dict]], top_k: int) -> List[Dict]:
    """Reciprocal Rank Fusion over multiple ranked lists keyed by ``_id``."""
    scored: Dict[str, Dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            hid = hit["_id"]
            entry = scored.setdefault(hid, {"hit": hit, "rrf": 0.0})
            entry["rrf"] += 1.0 / (_RRF_K + rank + 1)
    merged = sorted(scored.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
    out: List[Dict] = []
    for entry in merged:
        hit = entry["hit"]
        hit["_score"] = entry["rrf"]
        out.append(hit)
    return out


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    settings: Optional[RagSettings] = None,
) -> List[Passage]:
    cfg = settings or load_settings()
    client = get_client(cfg)
    embedder = Embedder(cfg)

    qvec = embedder.embed_query(query)
    bm25_hits = _bm25_search(client, cfg.index_name, query, candidate_k)
    knn_hits = _knn_search(client, cfg.index_name, qvec, candidate_k)
    merged = _rrf_merge([bm25_hits, knn_hits], top_k=top_k)

    passages: List[Passage] = []
    for hit in merged:
        src = hit.get("_source", {})
        passages.append(
            Passage(
                page_id=src.get("page_id", ""),
                title=src.get("title", "Untitled"),
                text=src.get("text", ""),
                url=src.get("url", ""),
                space_key=src.get("space_key", ""),
                source_name=src.get("source_name", ""),
                chunk_index=int(src.get("chunk_index", 0)),
                score=float(hit.get("_score", 0.0)),
            )
        )
    return passages


def format_passages_for_llm(passages: List[Passage]) -> str:
    """Human-readable string for LLM consumption."""
    if not passages:
        return "No relevant Confluence content found."
    parts = [f"Top {len(passages)} Confluence passages:\n"]
    for p in passages:
        header = f"### {p.title} (page_id={p.page_id}, score={p.score:.3f})"
        if p.url:
            header += f" — {p.url}"
        parts.append(header)
        parts.append(p.text)
        parts.append("")
    return "\n".join(parts)
