"""
FastAPI REST API for the dashboard AI chatbot.

Runs as a standalone service on Uvicorn, separate from the dashboard.
The dashboard calls this API over HTTP (``CHAT_API_URL`` env var).

    uvicorn ai_agent.chat_api:app --host 0.0.0.0 --port 8051

Endpoints
---------
POST /api/chat                       — Send a message, get AI response
POST /api/report/description         — LLM-generate one figure's description
POST /api/report/summary             — LLM-generate the overall report summary
GET  /api/metadata/columns           — List all column names and categories
GET  /api/metadata/columns/{name}    — Get details for one column
GET  /api/metadata/groups            — List all group definitions
GET  /api/health                     — Health check
POST /api/metadata/rebuild           — Force-rebuild the metadata cache
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai_agent.chat_agent import achat, init_metadata
from ai_agent.metadata_builder import build_metadata
from ai_agent.report_agent.description import (
    STATUS_APPENDED,
    STATUS_GENERATED,
    STATUS_NO_INSTRUCTIONS,
    STATUS_REGENERATED,
    generate_description,
    generate_summary,
)
from ai_agent.slack_handler import build_slack_handler

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
    provider: Optional[str] = Field(
        None,
        description="(Deprecated — use 'model' instead) LLM provider override. "
        "Falls back to CHAT_PROVIDER env var if not set.",
    )
    model: Optional[str] = Field(
        None,
        description="Model key from the catalog, e.g. 'openai:gpt-4.1' or 'gemini:gemini-2.5-pro'. "
        "Overrides 'provider' when set.",
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
# Report tab — description / summary generation
# ---------------------------------------------------------------------------

GenerateStatusLiteral = Literal["generated", "regenerated", "appended", "no_instructions"]

# Map the internal STATUS_* constants from description.py to the wire enum.
# The constants are user-facing strings ("Generated.", etc.) and may evolve;
# this layer is the stable contract.
_STATUS_TO_WIRE: Dict[str, GenerateStatusLiteral] = {
    STATUS_GENERATED: "generated",
    STATUS_REGENERATED: "regenerated",
    STATUS_APPENDED: "appended",
    STATUS_NO_INSTRUCTIONS: "no_instructions",
}


class ReferenceItem(BaseModel):
    url: str = Field(..., description="Original URL the user attached")
    title: Optional[str] = Field(None, description="Resolved page title; falls back to URL")
    text: Optional[str] = Field(None, description="Stripped, truncated body. Empty for external links.")
    error: Optional[str] = Field(None, description="Set when fetch failed; surfaced to the LLM as context")


class GenerateDescriptionRequest(BaseModel):
    data_summary: Dict[str, Any] = Field(
        ...,
        description="JSON output of dashboards.report_agent.data_summary.extract_data_summary(figure)",
    )
    existing_description: Optional[str] = Field(
        None,
        description="Current textarea contents — may include /prompt lines. None / '' triggers a first draft.",
    )
    references: List[ReferenceItem] = Field(
        default_factory=list,
        description=(
            "Pre-fetched references. The dashboard owns Confluence fetching "
            "(load_references) and ships the bodies here; ai_agent only "
            "inlines them into the prompt."
        ),
    )
    language: str = Field("en", description='"en", "zh-Hans", or "zh-Hant"')
    model: Optional[str] = Field(None, description='Optional model_key like "openai:gpt-4.1"')


class FigureForSummary(BaseModel):
    data_summary: Dict[str, Any]
    description: Optional[str] = None


class GenerateSummaryRequest(BaseModel):
    figures: List[FigureForSummary]
    existing_summary: Optional[str] = None
    references: List[ReferenceItem] = Field(default_factory=list)
    language: str = "en"
    model: Optional[str] = None


class GenerateResponse(BaseModel):
    text: str = Field(..., description="Full text to display in the textarea")
    status: GenerateStatusLiteral
    message: str = Field(..., description="Human-readable status hint for the UI")
    elapsed_ms: int


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

    @asynccontextmanager
    async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            init_metadata()
            logger.info("Chat API: metadata pre-warmed at startup")
        except Exception:
            logger.warning("Chat API: metadata pre-warm failed (will retry on first request)", exc_info=True)
        yield

    app = FastAPI(
        title="Dashboard AI Chat API",
        description="AI-powered Q&A for game analytics dashboard columns and metrics",
        version="1.0.0",
        lifespan=_lifespan,
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

    # --- Chat endpoint ---
    @app.post("/api/chat", response_model=ChatResponse)
    async def post_chat(req: ChatRequest) -> ChatResponse:
        t0 = time.monotonic()

        model_key = req.model
        if not model_key and req.provider:
            os.environ["CHAT_PROVIDER"] = req.provider

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
            response_text = await achat(
                user_message=req.message,
                history=history,
                dashboard_config_path=config_path,
                model_key=model_key,
            )
        except Exception as exc:
            logger.exception("Chat agent error")
            raise HTTPException(status_code=500, detail=str(exc))

        elapsed = int((time.monotonic() - t0) * 1000)
        return ChatResponse(response=response_text, elapsed_ms=elapsed)

    # --- Report tab endpoints ---
    @app.post("/api/report/description", response_model=GenerateResponse)
    async def post_report_description(req: GenerateDescriptionRequest) -> GenerateResponse:
        t0 = time.monotonic()
        try:
            text, status = generate_description(
                data_summary=req.data_summary,
                language=req.language,
                model_key=req.model,
                existing_description=req.existing_description,
                references=[r.model_dump(exclude_none=True) for r in req.references],
            )
        except Exception as exc:
            logger.exception("Report description generation failed")
            raise HTTPException(status_code=500, detail=str(exc))
        return GenerateResponse(
            text=text,
            status=_STATUS_TO_WIRE[status],
            message=status,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )

    @app.post("/api/report/summary", response_model=GenerateResponse)
    async def post_report_summary(req: GenerateSummaryRequest) -> GenerateResponse:
        t0 = time.monotonic()
        try:
            text, status = generate_summary(
                figures=[fig.model_dump(exclude_none=True) for fig in req.figures],
                language=req.language,
                model_key=req.model,
                existing_summary=req.existing_summary,
                references=[r.model_dump(exclude_none=True) for r in req.references],
            )
        except Exception as exc:
            logger.exception("Report summary generation failed")
            raise HTTPException(status_code=500, detail=str(exc))
        return GenerateResponse(
            text=text,
            status=_STATUS_TO_WIRE[status],
            message=status,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )

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

    @app.get("/health", response_model=HealthResponse)
    @app.get("/api/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        from ai_agent.chat_agent import _METADATA

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

    # --- Slack bot endpoint (only mounted when secrets are configured) ---
    slack_handler = build_slack_handler()
    if slack_handler is not None:

        @app.post("/slack/events")
        async def slack_events(req: Request):  # type: ignore[no-untyped-def]
            return await slack_handler.handle(req)

        logger.info("Slack bot mounted at /slack/events")

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by Uvicorn in production and local dev)
#   uvicorn ai_agent.chat_api:app --host 0.0.0.0 --port 8051
# ---------------------------------------------------------------------------
app = create_chat_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8051, log_level="info")


if __name__ == "__main__":
    main()
