"""
Faiss-based vector store for the RAG service (Phase 1 backend).

Persists a flat inner-product Faiss index (cosine similarity over
L2-normalized vectors) plus a JSON metadata sidecar to either a local
directory or an S3 prefix. The indexer rebuilds and re-uploads the pair;
the FastAPI service loads them at startup and keeps them in memory.

Artifact layout under ``{index_uri}``:

    confluence_rag.faiss        binary Faiss index
    confluence_rag.meta.json    chunk metadata, one entry per vector
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_FAISS_FILENAME = "confluence_rag.faiss"
_META_FILENAME = "confluence_rag.meta.json"


@dataclass
class StoredChunk:
    """One row of metadata, parallel to one row in the Faiss index."""

    page_id: str
    chunk_index: int
    title: str
    text: str
    url: str
    space_key: str
    source_name: str
    updated_at: str


def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def _split_s3(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


class FaissStore:
    """Flat-IP Faiss index keyed by row order to ``self.chunks``."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index: Any = None
        self.chunks: List[StoredChunk] = []

    @classmethod
    def build(
        cls,
        items: List[Tuple[StoredChunk, List[float]]],
        dim: int,
    ) -> "FaissStore":
        import faiss
        import numpy as np

        store = cls(dim)
        store.index = faiss.IndexFlatIP(dim)
        if not items:
            return store

        vectors = np.asarray([v for _, v in items], dtype=np.float32)
        if vectors.shape[1] != dim:
            raise ValueError(f"Embedding dim mismatch: got {vectors.shape[1]}, want {dim}")
        # Cosine via inner product on L2-normalized vectors.
        faiss.normalize_L2(vectors)
        store.index.add(vectors)
        store.chunks = [c for c, _ in items]
        return store

    def search(self, query_vec: List[float], top_k: int) -> List[Tuple[StoredChunk, float]]:
        import faiss
        import numpy as np

        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.asarray([query_vec], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        out: List[Tuple[StoredChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            out.append((self.chunks[idx], float(score)))
        return out

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, uri: str) -> None:
        """Write the index + metadata to a local dir or s3:// prefix."""
        import faiss

        if self.index is None:
            raise RuntimeError("FaissStore has no index to save.")
        with tempfile.TemporaryDirectory() as tmp:
            faiss_path = Path(tmp) / _FAISS_FILENAME
            meta_path = Path(tmp) / _META_FILENAME
            faiss.write_index(self.index, str(faiss_path))
            meta = {
                "dim": self.dim,
                "count": len(self.chunks),
                "chunks": [asdict(c) for c in self.chunks],
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

            if is_s3_uri(uri):
                bucket, prefix = _split_s3(uri)
                _s3_upload(faiss_path, bucket, _join_s3_key(prefix, _FAISS_FILENAME))
                _s3_upload(meta_path, bucket, _join_s3_key(prefix, _META_FILENAME))
            else:
                target = Path(uri)
                target.mkdir(parents=True, exist_ok=True)
                (target / _FAISS_FILENAME).write_bytes(faiss_path.read_bytes())
                (target / _META_FILENAME).write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Saved FaissStore to %s (%d vectors)", uri, len(self.chunks))

    @classmethod
    def load(cls, uri: str) -> "FaissStore":
        """Read the index + metadata from a local dir or s3:// prefix."""
        import faiss

        with tempfile.TemporaryDirectory() as tmp:
            faiss_local = Path(tmp) / _FAISS_FILENAME
            meta_local = Path(tmp) / _META_FILENAME
            if is_s3_uri(uri):
                bucket, prefix = _split_s3(uri)
                _s3_download(bucket, _join_s3_key(prefix, _FAISS_FILENAME), faiss_local)
                _s3_download(bucket, _join_s3_key(prefix, _META_FILENAME), meta_local)
            else:
                src = Path(uri)
                faiss_local.write_bytes((src / _FAISS_FILENAME).read_bytes())
                meta_local.write_text((src / _META_FILENAME).read_text(encoding="utf-8"), encoding="utf-8")

            index = faiss.read_index(str(faiss_local))
            meta = json.loads(meta_local.read_text(encoding="utf-8"))

        store = cls(int(meta["dim"]))
        store.index = index
        store.chunks = [StoredChunk(**c) for c in meta.get("chunks") or []]
        if store.index.ntotal != len(store.chunks):
            logger.warning(
                "Faiss/metadata size mismatch at %s: index=%d, meta=%d",
                uri,
                store.index.ntotal,
                len(store.chunks),
            )
        return store


def _join_s3_key(prefix: str, filename: str) -> str:
    prefix = (prefix or "").rstrip("/")
    return f"{prefix}/{filename}" if prefix else filename


def _s3_upload(local_path: Path, bucket: str, key: str) -> None:
    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)


def _s3_download(bucket: str, key: str, local_path: Path) -> None:
    import boto3

    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))
