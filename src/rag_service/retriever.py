"""
Query-time retrieval, dispatched by ``RagSettings.backend``.

- ``faiss`` (Phase 1): pure dense similarity over the in-memory Faiss
  store loaded once from local disk or S3 and cached at module level.
- ``opensearch`` (Phase 2): BM25 + vector kNN merged with Reciprocal
  Rank Fusion (RRF). RRF works on every OpenSearch flavor including
  Serverless, where the ``hybrid`` query plugin may not be available.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from rag_service.config import RagSettings, load_settings
from rag_service.embeddings import Embedder

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

# Module-level Faiss store cache. Loaded lazily on first retrieve and
# refreshed via reload_faiss_store() after a rebuild.
_faiss_store: Optional[object] = None
_faiss_store_uri: Optional[str] = None
_faiss_lock = threading.Lock()


def reload_faiss_store(settings: Optional[RagSettings] = None) -> None:
    """Force a reload of the cached Faiss store (call after a rebuild)."""
    global _faiss_store, _faiss_store_uri
    cfg = settings or load_settings()
    with _faiss_lock:
        _faiss_store = None
        _faiss_store_uri = cfg.index_uri


def _get_faiss_store(uri: str):
    global _faiss_store, _faiss_store_uri
    if _faiss_store is not None and _faiss_store_uri == uri:
        return _faiss_store
    with _faiss_lock:
        if _faiss_store is None or _faiss_store_uri != uri:
            from rag_service.faiss_store import FaissStore

            _faiss_store = FaissStore.load(uri)
            _faiss_store_uri = uri
    return _faiss_store


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
    if cfg.backend == "faiss":
        return _retrieve_faiss(cfg, query, top_k=top_k)
    return _retrieve_opensearch(cfg, query, top_k=top_k, candidate_k=candidate_k)


def _retrieve_faiss(cfg: RagSettings, query: str, *, top_k: int) -> List[Passage]:
    embedder = Embedder(cfg)
    qvec = embedder.embed_query(query)
    store = _get_faiss_store(cfg.index_uri)
    hits = store.search(qvec, top_k=top_k)  # type: ignore[union-attr]
    return [
        Passage(
            page_id=c.page_id,
            title=c.title,
            text=c.text,
            url=c.url,
            space_key=c.space_key,
            source_name=c.source_name,
            chunk_index=c.chunk_index,
            score=score,
        )
        for c, score in hits
    ]


def _retrieve_opensearch(cfg: RagSettings, query: str, *, top_k: int, candidate_k: int) -> List[Passage]:
    from rag_service.opensearch_client import get_client

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
