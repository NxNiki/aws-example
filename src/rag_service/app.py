"""
FastAPI service for the RAG microservice.

Run locally:
    uvicorn rag_service.app:app --host 0.0.0.0 --port 8052

Endpoints:
    GET  /health                     liveness check + index status
    GET  /sources                    show what rag_sources.yaml lists
    POST /retrieve                   {query, top_k} → ranked passages

Index (re)builds are NOT served here — they run out-of-band via
``jobs/build_rag_index.py`` (daily Fargate task, see
infra/rag_service/setup_reindex_schedule.sh) so a multi-minute embed
never blocks this service's event loop. Locally, run that script directly.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_service.config import RagSettings, load_settings
from rag_service.embeddings import resolved_embedding_dim
from rag_service.retriever import Passage, retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural-language question")
    top_k: int = Field(5, ge=1, le=20)
    candidate_k: int = Field(20, ge=5, le=100)


class PassageOut(BaseModel):
    page_id: str
    title: str
    text: str
    url: str
    space_key: str
    source_name: str
    chunk_index: int
    score: float


class RetrieveResponse(BaseModel):
    query: str
    passages: List[PassageOut]
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    index: str
    index_exists: bool
    doc_count: Optional[int] = None


class SourceOut(BaseModel):
    name: str
    space_key: Optional[str]
    page_ids: List[str]
    include_children: bool


class SourcesResponse(BaseModel):
    sources: List[SourceOut]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    settings: RagSettings = load_settings()

    @asynccontextmanager
    async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
        if settings.backend == "faiss":
            try:
                from rag_service.faiss_store import FaissStore

                store = FaissStore.load(settings.index_uri)
                logger.info(
                    "RAG service ready: backend=faiss uri=%s vectors=%d",
                    settings.index_uri,
                    store.ntotal,
                )
            except Exception:
                logger.warning(
                    "FaissStore load failed at startup (uri=%s) — run build_rag_index.py",
                    settings.index_uri,
                    exc_info=True,
                )
        else:
            try:
                from rag_service.opensearch_client import ensure_index, get_client

                client = get_client(settings)
                ensure_index(client, settings.index_name, resolved_embedding_dim(settings))
                logger.info("RAG service ready: backend=opensearch index=%s", settings.index_name)
            except Exception:
                logger.warning("Index ensure failed at startup", exc_info=True)
        yield

    app = FastAPI(
        title="Confluence RAG Service",
        description="Vector retrieval over Confluence documentation (Faiss / OpenSearch)",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        if settings.backend == "faiss":
            from rag_service.faiss_store import FaissStore

            try:
                store = FaissStore.load(settings.index_uri)
                return HealthResponse(
                    status="ok",
                    index=settings.index_uri,
                    index_exists=True,
                    doc_count=store.ntotal,
                )
            except Exception:
                return HealthResponse(
                    status="ok",
                    index=settings.index_uri,
                    index_exists=False,
                    doc_count=None,
                )

        from rag_service.opensearch_client import get_client

        client = get_client(settings)
        exists = client.indices.exists(index=settings.index_name)
        count: Optional[int] = None
        if exists:
            try:
                count = client.count(index=settings.index_name).get("count")
            except Exception:
                count = None
        return HealthResponse(
            status="ok",
            index=settings.index_name,
            index_exists=bool(exists),
            doc_count=count,
        )

    @app.get("/sources", response_model=SourcesResponse)
    async def sources() -> SourcesResponse:
        return SourcesResponse(
            sources=[
                SourceOut(
                    name=s.name,
                    space_key=s.space_key,
                    page_ids=s.page_ids,
                    include_children=s.include_children,
                )
                for s in settings.sources
            ]
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def post_retrieve(req: RetrieveRequest) -> RetrieveResponse:
        t0 = time.monotonic()
        try:
            passages: List[Passage] = retrieve(
                req.query,
                top_k=req.top_k,
                candidate_k=req.candidate_k,
                settings=settings,
            )
        except Exception as exc:
            logger.exception("Retrieve failed")
            raise HTTPException(status_code=500, detail=str(exc))
        elapsed = int((time.monotonic() - t0) * 1000)
        return RetrieveResponse(
            query=req.query,
            passages=[PassageOut(**p.__dict__) for p in passages],
            elapsed_ms=elapsed,
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    cfg = load_settings()
    uvicorn.run(app, host="0.0.0.0", port=cfg.service_port, log_level="info")


if __name__ == "__main__":
    main()
