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

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from ai_agent.chat_agent import (
    _build_messages,
    _ensure_metadata,
    achat,
    create_agent,
    init_metadata,
)
from ai_agent.dashboard_actions import ACTION_TOOL_NAMES, ACTION_TOOLS, CLARIFY_TOOL_NAME
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


class AgentChatRequest(ChatRequest):
    """Request for the streaming /api/agent/chat endpoint (the React dashboard).

    Extends ChatRequest with a snapshot of the dashboard's UI state so the
    agent can drive it with action tools (which config/tab is active, which
    panels/metrics exist, current date windows).
    """

    dashboard_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque UI-state snapshot from the React dashboard store",
    )


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
        description="JSON summary of the figure's series/axes (built client-side and POSTed here)",
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

# Where dashboard_config-*.yaml live. Env-driven (the images set
# DASHBOARD_CONFIG_DIR); defaults to the repo's configs/dashboard for local
# runs. Was Path(__file__).parent before, where no YAMLs ever existed — the
# agent silently dropped every dashboard_config request.
_DASHBOARD_DIR = Path(
    os.environ.get("DASHBOARD_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "configs" / "dashboard"))
)


def _sse(event: str, data: Dict[str, Any]) -> str:
    """One Server-Sent-Events frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _chunk_text(content: Any) -> str:
    """Plain text of one STREAMED chunk, whitespace preserved.

    Unlike ``_stringify_ai_content`` (which strips — fine for a final message),
    chunks are mid-word fragments: stripping them deletes the spaces/newlines at
    chunk boundaries ("RTP group" + " assignment" → "RTP groupassignment").
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if t is not None:
                    out.append(str(t))
        return "".join(out)
    return str(content)


# Instructions merged into the system prompt when the agent has the
# dashboard-action tools. Kept separate from the main system prompt so /api/chat
# and Slack behavior is unchanged.
_ACTION_GUIDE = """You are embedded in the game-stats dashboard and can OPERATE it with these tools:
load_config, navigate_tab, set_metrics, set_date_range, set_granularity, ask_user,
and the Report tab tools: set_report_period, regenerate_report, load_report_spec,
save_report_spec, add_report_figure, patch_report_figure, remove_report_figure.

Rules:
- When the user asks to SEE data (a metric, a comparison, a trend), drive the
  dashboard: switch config/tab if needed, then set the right panel's metrics.
  Choose the panel whose available metrics contain the requested metric.
- Use ONLY config ids, tab names, panel ids, and metric names that appear in
  the DASHBOARD STATE below. Never invent metric names — if unsure, check the
  panels' available metrics, or use the metadata tools.
- When the user asks to update/edit the REPORT, follow the matching skill in
  the SKILLS playbook provided in the message context. Report figure ids, saved
  spec names, and current sources are in the DASHBOARD STATE's report section.
- If the request is ambiguous (which game? by date or by group? which metric
  variant?), call ask_user with 2-4 concrete options, then END your turn.
- After dispatching actions, reply with one short sentence describing what you
  changed. Keep prose minimal when actions carry the answer.
- For pure knowledge questions (what does a column mean, how is RTP computed),
  just answer — no dashboard actions.
"""

_skills_cache: tuple[float, str] = (0.0, "")


def load_skills_markdown() -> str:
    """Load the agent skills registry (skills.md), mtime-cached.

    API endpoint: GET /api/agent/skills — multi-step procedures (update_report,
    show_metric, edit_report_figure) injected into the agent's context and
    served raw for introspection. Editing skills.md takes effect on the next
    chat turn without a restart.
    """
    global _skills_cache
    path = Path(__file__).resolve().parent / "skills.md"
    try:
        mtime = path.stat().st_mtime
        if mtime != _skills_cache[0]:
            _skills_cache = (mtime, path.read_text(encoding="utf-8"))
    except OSError:
        logger.warning("skills.md not readable at %s", path)
        return ""
    return _skills_cache[1]


def _agent_context_message(dashboard_state: Dict[str, Any]) -> SystemMessage:
    state_json = json.dumps(dashboard_state or {}, default=str)
    if len(state_json) > 12000:  # keep the prompt bounded; panels carry the bulk
        state_json = state_json[:12000] + "…(truncated)"
    return SystemMessage(content=f"{_ACTION_GUIDE}\n\nDASHBOARD STATE (current UI):\n{state_json}")


def _with_skills(message: str) -> str:
    """Prepend the skills playbook to the user turn as a labeled context block.

    The skills text must NOT go into the system instruction: gemini-2.5-flash
    deterministically returns an empty response (finish_reason STOP, zero
    output tokens, no tool calls) when instructional skill text rides in the
    system prompt — even at ~1.5k chars — while the same text in the human
    message works every time. Verified empirically; same bug family as the
    two-SystemMessage and replayed-AIMessage empties handled below.
    """
    skills = load_skills_markdown()
    if not skills:
        return message
    return (
        "[CONTEXT — agent skills playbook, not part of the user's message]\n"
        f"{skills}\n[END CONTEXT]\n\nUser request: {message}"
    )


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

    # --- Streaming agent endpoint (React dashboard chat panel) ---
    @app.post("/api/agent/chat")
    async def post_agent_chat(req: AgentChatRequest) -> StreamingResponse:
        """API endpoint: POST /api/agent/chat — streaming agent turn (SSE).

        Powers the React dashboard's chat panel. Streams the agent run as
        Server-Sent Events: ``token`` (text chunks), ``tool_start``/``tool_end``
        (knowledge tools), ``action`` (dashboard-driving tool calls the frontend
        dispatcher applies), ``clarify`` (quick-reply question), then ``done`` or
        ``error``. The blocking /api/chat stays for the legacy Dash app + Slack.
        """
        config_path: Optional[Path] = None
        if req.dashboard_config:
            candidate = _DASHBOARD_DIR / req.dashboard_config
            if candidate.exists():
                config_path = candidate

        history = [{"role": m.role, "content": m.content} for m in req.history]
        model_key = req.model

        def _merge_system(messages: List[Any]) -> List[Any]:
            # MERGE the action contract + UI snapshot into the base system
            # prompt. Gemini supports only ONE system instruction — a second
            # SystemMessage makes gemini-2.5-flash return an empty response
            # (finish_reason STOP, no content, no tool calls).
            ctx = _agent_context_message(req.dashboard_state)
            messages[0] = SystemMessage(content=f"{messages[0].content}\n\n{ctx.content}")
            return messages

        async def run_stream(agent: Any, messages: List[Any]) -> AsyncIterator[tuple[str, bool]]:
            """Yield (sse_frame, meaningful) for one agent run; `meaningful` marks
            frames that prove the model actually produced something."""
            async for ev in agent.astream_events({"messages": messages}, version="v2"):
                kind = ev.get("event")
                data = ev.get("data") or {}
                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    text = _chunk_text(getattr(chunk, "content", "")) if chunk is not None else ""
                    if text:
                        yield _sse("token", {"text": text}), True
                elif kind == "on_tool_start":
                    name = ev.get("name", "")
                    tool_input = data.get("input") or {}
                    if not isinstance(tool_input, dict):
                        tool_input = {"input": str(tool_input)}
                    if name == CLARIFY_TOOL_NAME:
                        yield _sse(
                            "clarify",
                            {
                                "question": tool_input.get("question", ""),
                                "options": tool_input.get("options") or [],
                            },
                        ), True
                    elif name in ACTION_TOOL_NAMES:
                        yield _sse("action", {"type": name, **tool_input}), True
                    else:
                        yield _sse(
                            "tool_start", {"name": name, "input": json.dumps(tool_input, default=str)[:400]}
                        ), True
                elif kind == "on_tool_end":
                    name = ev.get("name", "")
                    if name not in ACTION_TOOL_NAMES:
                        yield _sse("tool_end", {"name": name, "output": str(data.get("output", ""))[:400]}), False

        async def gen() -> AsyncIterator[str]:
            t0 = time.monotonic()
            try:
                _ensure_metadata(dashboard_config_path=config_path)
                agent = create_agent(model_key=model_key, extra_tools=ACTION_TOOLS)

                produced = False
                messages = _merge_system(_build_messages(_with_skills(req.message), history))
                async for frame, meaningful in run_stream(agent, messages):
                    produced = produced or meaningful
                    yield frame

                if not produced and history:
                    # Gemini sometimes returns a completely empty response when
                    # its own long answer is replayed as an AIMessage (content-
                    # dependent). Retry once with the conversation folded into a
                    # plain transcript, which sidesteps the replay path.
                    logger.warning("Agent stream produced nothing; retrying with transcript history")
                    transcript = "\n".join(
                        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history
                    )
                    folded = f"Previous conversation:\n{transcript}\n\nNew user message: {req.message}"
                    messages = _merge_system(_build_messages(_with_skills(folded), []))
                    async for frame, meaningful in run_stream(agent, messages):
                        produced = produced or meaningful
                        yield frame

                if not produced:
                    yield _sse(
                        "error", {"message": "The model returned an empty response — please try again or rephrase."}
                    )
                yield _sse("done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)})
            except Exception as exc:
                logger.exception("Agent stream error")
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            # no-cache + X-Accel-Buffering keep proxies from buffering the stream.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/agent/skills")
    async def get_agent_skills() -> Dict[str, str]:
        """API endpoint: GET /api/agent/skills — the skills.md registry, raw.

        Introspection only: the agent reads the file in-process at prompt-build
        time; this endpoint lets the UI / operators see what the agent was given.
        """
        return {"markdown": load_skills_markdown()}

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
