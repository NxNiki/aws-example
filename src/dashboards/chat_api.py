"""
FastAPI REST API for the dashboard AI chatbot.

Deployment modes
----------------
1. **Embedded (current)**: The Dash/Gunicorn server in game_stats_monitor.py
   registers equivalent Flask blueprint routes at ``/api/*``.  No separate
   process needed — everything runs in one ECS task.

2. **Standalone (future React frontend)**: Run this module directly with
   Uvicorn (``python -m dashboards.chat_api`` or
   ``uvicorn dashboards.chat_api:app``).  Ideal when the frontend is a
   separate React/Next.js app that talks to this API over HTTP.

Endpoints
---------
POST /api/chat          — Send a message, get AI response
GET  /api/metadata/columns          — List all column names and categories
GET  /api/metadata/columns/{name}   — Get details for one column
GET  /api/metadata/groups           — List all group definitions
GET  /api/health                    — Health check
POST /api/metadata/rebuild          — Force-rebuild the metadata cache
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dashboards.chat_agent import chat, init_metadata
from dashboards.metadata_builder import build_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models (request / response schemas)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's question")
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages for multi-turn context",
    )
    dashboard_config: Optional[str] = Field(
        None,
        description="Filename of the active dashboard config (e.g. 'dashboard_config-ss01.yaml')",
    )


class ChatResponse(BaseModel):
    response: str
    elapsed_ms: int = Field(..., description="Server-side processing time in milliseconds")


class ColumnInfo(BaseModel):
    name: str
    category: str
    description: str
    formula: Optional[str] = None
    etl_formula: Optional[Dict[str, str]] = None


class ColumnListResponse(BaseModel):
    columns: List[ColumnInfo]
    total: int


class GroupValueInfo(BaseModel):
    name: str
    definition: Optional[str] = None
    description: str


class GroupInfo(BaseModel):
    name: str
    description: str
    values: List[GroupValueInfo]


class GroupListResponse(BaseModel):
    groups: List[GroupInfo]


class HealthResponse(BaseModel):
    status: str
    metadata_loaded: bool


class RebuildResponse(BaseModel):
    status: str
    columns_count: int
    etl_sources_count: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_DASHBOARD_DIR = Path(__file__).resolve().parent


def create_chat_app(*, prefix: str = "") -> FastAPI:
    """
    Create the FastAPI app for the chat API.

    Parameters
    ----------
    prefix : str
        URL prefix (e.g. "/chat-api") when mounting under another ASGI app.
    """
    app = FastAPI(
        title="Dashboard AI Chat API",
        description="AI-powered Q&A for game analytics dashboard columns and metrics",
        version="1.0.0",
        docs_url=f"{prefix}/docs" if prefix else "/docs",
        openapi_url=f"{prefix}/openapi.json" if prefix else "/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            init_metadata()
            logger.info("Chat API: metadata pre-warmed at startup")
        except Exception:
            logger.warning("Chat API: metadata pre-warm failed (will retry on first request)", exc_info=True)

    # --- Chat endpoint ---
    @app.post("/api/chat", response_model=ChatResponse)
    async def post_chat(req: ChatRequest) -> ChatResponse:
        t0 = time.monotonic()

        config_path: Optional[Path] = None
        if req.dashboard_config:
            config_path = _DASHBOARD_DIR / req.dashboard_config
            if not config_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Dashboard config '{req.dashboard_config}' not found",
                )

        history = [{"role": m.role, "content": m.content} for m in req.history]

        try:
            response_text = chat(
                user_message=req.message,
                history=history,
                dashboard_config_path=config_path,
            )
        except Exception as exc:
            logger.exception("Chat agent error")
            raise HTTPException(status_code=500, detail=str(exc))

        elapsed = int((time.monotonic() - t0) * 1000)
        return ChatResponse(response=response_text, elapsed_ms=elapsed)

    # --- Metadata endpoints ---
    @app.get("/api/metadata/columns", response_model=ColumnListResponse)
    async def list_columns(category: Optional[str] = None) -> ColumnListResponse:
        meta = build_metadata()
        columns = meta.get("columns", {})
        items: List[ColumnInfo] = []
        for name, info in sorted(columns.items()):
            if category and info.get("category") != category:
                continue
            etl = info.get("etl_formula")
            if isinstance(etl, str):
                etl = {"note": etl}
            items.append(
                ColumnInfo(
                    name=name,
                    category=info.get("category", "other"),
                    description=info.get("description", ""),
                    formula=info.get("formula"),
                    etl_formula=etl if isinstance(etl, dict) else None,
                )
            )
        return ColumnListResponse(columns=items, total=len(items))

    @app.get("/api/metadata/columns/{name}", response_model=ColumnInfo)
    async def get_column(name: str) -> ColumnInfo:
        meta = build_metadata()
        columns = meta.get("columns", {})
        if name not in columns:
            raise HTTPException(status_code=404, detail=f"Column '{name}' not found")
        info = columns[name]
        etl = info.get("etl_formula")
        if isinstance(etl, str):
            etl = {"note": etl}
        return ColumnInfo(
            name=name,
            category=info.get("category", "other"),
            description=info.get("description", ""),
            formula=info.get("formula"),
            etl_formula=etl if isinstance(etl, dict) else None,
        )

    @app.get("/api/metadata/groups", response_model=GroupListResponse)
    async def list_groups() -> GroupListResponse:
        meta = build_metadata()
        groups_data = meta.get("groups", {})
        groups: List[GroupInfo] = []
        for gname, ginfo in groups_data.items():
            values = [
                GroupValueInfo(
                    name=vname,
                    definition=vinfo.get("definition"),
                    description=vinfo.get("description", ""),
                )
                for vname, vinfo in ginfo.get("values", {}).items()
            ]
            groups.append(
                GroupInfo(
                    name=gname,
                    description=ginfo.get("description", ""),
                    values=values,
                )
            )
        return GroupListResponse(groups=groups)

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        from dashboards.chat_agent import _METADATA

        return HealthResponse(status="ok", metadata_loaded=_METADATA is not None)

    @app.post("/api/metadata/rebuild", response_model=RebuildResponse)
    async def rebuild_metadata() -> RebuildResponse:
        meta = build_metadata(force_rebuild=True)
        init_metadata(force_rebuild=True)
        return RebuildResponse(
            status="rebuilt",
            columns_count=len(meta.get("columns", {})),
            etl_sources_count=len(meta.get("etl_sources", [])),
        )

    return app


# ---------------------------------------------------------------------------
# Standalone entry point: ``python -m dashboards.chat_api``
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    app = create_chat_app()
    uvicorn.run(app, host="0.0.0.0", port=8051, log_level="info")


if __name__ == "__main__":
    main()
