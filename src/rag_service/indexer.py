"""
Build / refresh the OpenSearch index from rag_sources.yaml.

Pipeline:
  load sources  →  fetch Confluence pages  →  chunk  →  embed  →  bulk index

Idempotent: each chunk's _id is ``{page_id}-{chunk_index}`` so re-running
upserts in place rather than duplicating.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from opensearchpy.helpers import bulk

from rag_service.chunker import Chunk, chunk_text
from rag_service.config import RagSettings, load_settings
from rag_service.confluence_loader import iter_source_pages
from rag_service.embeddings import Embedder, resolved_embedding_dim
from rag_service.opensearch_client import ensure_index, get_client

logger = logging.getLogger(__name__)


def _gather_chunks(settings: RagSettings) -> List[Chunk]:
    chunks: List[Chunk] = []
    for source in settings.sources:
        logger.info("Loading source: %s", source.name)
        pages_seen = 0
        for page_id, title, text, url, space_key in iter_source_pages(source):
            pages_seen += 1
            chunks.extend(
                chunk_text(
                    page_id=page_id,
                    title=title,
                    text=text,
                    url=url,
                    space_key=space_key,
                    source_name=source.name,
                    chunk_chars=settings.chunk_chars,
                    overlap=settings.chunk_overlap,
                )
            )
        logger.info("  → %s: %d pages collected", source.name, pages_seen)
    return chunks


def _to_action(chunk: Chunk, embedding: List[float], index_name: str, ts: str) -> Dict:
    return {
        "_op_type": "index",
        "_index": index_name,
        "_id": f"{chunk.page_id}-{chunk.chunk_index}",
        "_source": {
            "page_id": chunk.page_id,
            "title": chunk.title,
            "text": chunk.text,
            "url": chunk.url,
            "space_key": chunk.space_key,
            "source_name": chunk.source_name,
            "chunk_index": chunk.chunk_index,
            "updated_at": ts,
            "embedding": embedding,
        },
    }


def build_index(settings: Optional[RagSettings] = None, batch_size: int = 200) -> Dict[str, int]:
    """Run the full pipeline. Returns a counts summary."""
    cfg = settings or load_settings()
    client = get_client(cfg)
    dim = resolved_embedding_dim(cfg)
    ensure_index(client, cfg.index_name, dim)

    chunks = _gather_chunks(cfg)
    if not chunks:
        logger.warning("No chunks produced — check rag_sources.yaml.")
        return {"pages": 0, "chunks": 0, "indexed": 0}

    embedder = Embedder(cfg)
    ts = datetime.now(timezone.utc).isoformat()

    indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        actions = [_to_action(chunk, vec, cfg.index_name, ts) for chunk, vec in zip(batch, vectors)]
        success, errors = bulk(client, actions, raise_on_error=False)
        indexed += success
        if errors:
            logger.warning("Bulk batch had %d errors (first: %s)", len(errors), errors[0])
        logger.info("Indexed %d / %d chunks", indexed, len(chunks))

    pages = len({c.page_id for c in chunks})
    return {"pages": pages, "chunks": len(chunks), "indexed": indexed}


def delete_index(settings: Optional[RagSettings] = None) -> None:
    """Drop the index — use before a clean rebuild if the schema changes."""
    cfg = settings or load_settings()
    client = get_client(cfg)
    if client.indices.exists(cfg.index_name):
        client.indices.delete(cfg.index_name)
        logger.info("Deleted index: %s", cfg.index_name)
