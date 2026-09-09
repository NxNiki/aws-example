"""
Load runtime configuration for the RAG service.

Two layers:
- ``rag_sources.yaml`` — what to index (Confluence spaces, page IDs, URLs).
  Lives in ``src/rag_service/config/rag_sources.yaml`` so it's bundled with
  the service image but can be overridden by the ``RAG_SOURCES_PATH`` env var
  for tests or local experimentation.
- Env vars — how to connect (OpenSearch host, auth mode, embedding model).

Keep both knobs separate: the YAML lists *content*, env vars list
*infrastructure*.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from bituslabs_ds.config import DEFAULT_RAG_INDEX_URI
from bituslabs_ds.confluence.client import extract_folder_ids_from_urls, extract_page_ids_from_urls

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_SOURCES_PATH = _THIS_DIR / "config" / "rag_sources.yaml"


@dataclass
class ConfluenceSource:
    """One entry in rag_sources.yaml describing a chunk of Confluence content."""

    name: str
    space_key: Optional[str] = None
    page_ids: List[str] = field(default_factory=list)
    page_urls: List[str] = field(default_factory=list)
    folder_ids: List[str] = field(default_factory=list)
    folder_urls: List[str] = field(default_factory=list)
    include_children: bool = False
    labels: List[str] = field(default_factory=list)
    exclude_page_ids: List[str] = field(default_factory=list)


@dataclass
class RagSettings:
    """All runtime settings, derived from env vars + the sources YAML."""

    # Backend selection
    backend: str  # 'faiss' (default) | 'opensearch'
    index_uri: str  # local dir or s3:// prefix for the Faiss artifact

    # Indexing
    index_name: str
    embedding_provider: str  # 'openai' | 'gemini' | 'auto'
    embedding_model: Optional[str]  # None → use provider default
    embedding_dim: int  # may be the resolved default if not pinned
    embedding_dim_explicit: bool  # True if RAG_EMBEDDING_DIM was set
    chunk_chars: int
    chunk_overlap: int

    # OpenSearch connection (Phase 2 only — ignored when backend='faiss')
    opensearch_host: str
    opensearch_port: int
    opensearch_use_ssl: bool
    opensearch_auth_mode: str  # 'basic' | 'aws'
    opensearch_user: Optional[str]
    opensearch_password: Optional[str]
    aws_region: Optional[str]
    aws_service: str  # 'aoss' (Serverless) or 'es' (managed)

    # Service
    service_port: int
    sources: List[ConfluenceSource]


def load_sources(path: Optional[Path] = None) -> List[ConfluenceSource]:
    sources_path = path or Path(os.environ.get("RAG_SOURCES_PATH", _DEFAULT_SOURCES_PATH))
    if not sources_path.exists():
        raise FileNotFoundError(f"rag_sources file not found: {sources_path}")

    with open(sources_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    sources_raw = raw.get("sources") or []
    sources: List[ConfluenceSource] = []
    for entry in sources_raw:
        # Auto-classify URLs into pages vs folders so users don't need to know
        # the difference — they can just paste anything from the Confluence
        # browser into ``page_urls``. ``folder_urls`` is also accepted for
        # explicitness.
        page_urls_raw = list(entry.get("page_urls") or [])
        folder_urls_raw = list(entry.get("folder_urls") or [])

        page_ids = list(entry.get("page_ids") or [])
        page_ids.extend(extract_page_ids_from_urls(page_urls_raw))

        folder_ids = [str(f) for f in (entry.get("folder_ids") or [])]
        folder_ids.extend(extract_folder_ids_from_urls(page_urls_raw))
        folder_ids.extend(extract_folder_ids_from_urls(folder_urls_raw))

        sources.append(
            ConfluenceSource(
                name=entry["name"],
                space_key=entry.get("space_key"),
                page_ids=sorted(set(page_ids)),
                page_urls=page_urls_raw,
                folder_ids=sorted(set(folder_ids)),
                folder_urls=folder_urls_raw,
                include_children=bool(entry.get("include_children", False)),
                labels=list(entry.get("labels") or []),
                exclude_page_ids=[str(p) for p in (entry.get("exclude_page_ids") or [])],
            )
        )
    return sources


def load_settings(sources_path: Optional[Path] = None) -> RagSettings:
    sources = load_sources(sources_path)

    # Provider-aware defaults for model/dim. Resolution that depends on which
    # API keys are available happens later in embeddings.Embedder.
    provider = (os.environ.get("RAG_EMBEDDING_PROVIDER") or "auto").lower()
    explicit_dim_raw = os.environ.get("RAG_EMBEDDING_DIM")
    explicit_dim = explicit_dim_raw is not None
    if explicit_dim_raw is not None:
        embedding_dim = int(explicit_dim_raw)
    elif provider == "gemini":
        embedding_dim = 768  # gemini-embedding-001 truncated via Matryoshka
    elif provider == "openai":
        embedding_dim = 1536  # text-embedding-3-small
    else:
        # Auto: mirror Embedder._resolve_provider's preference (OpenAI first,
        # else Gemini) so the default dim matches the provider that will
        # actually be used.
        if os.environ.get("OPENAI_API_KEY"):
            embedding_dim = 1536
        elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            embedding_dim = 768
        else:
            embedding_dim = 1536

    return RagSettings(
        backend=(os.environ.get("RAG_BACKEND") or "faiss").lower(),
        index_uri=DEFAULT_RAG_INDEX_URI,
        index_name=os.environ.get("OPENSEARCH_INDEX", "confluence_rag"),
        embedding_provider=provider,
        embedding_model=os.environ.get("RAG_EMBEDDING_MODEL"),
        embedding_dim=embedding_dim,
        embedding_dim_explicit=explicit_dim,
        chunk_chars=int(os.environ.get("RAG_CHUNK_CHARS", "1200")),
        chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", "200")),
        opensearch_host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        opensearch_port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
        opensearch_use_ssl=os.environ.get("OPENSEARCH_SSL", "true").lower() == "true",
        opensearch_auth_mode=os.environ.get("OPENSEARCH_AUTH", "basic").lower(),
        opensearch_user=os.environ.get("OPENSEARCH_USER", "admin"),
        opensearch_password=os.environ.get("OPENSEARCH_PASSWORD"),
        aws_region=os.environ.get("AWS_REGION", "us-west-2"),
        aws_service=os.environ.get("OPENSEARCH_SERVICE", "aoss"),
        service_port=int(os.environ.get("RAG_SERVICE_PORT", "8052")),
        sources=sources,
    )
