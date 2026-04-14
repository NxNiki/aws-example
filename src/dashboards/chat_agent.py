"""
LangChain-based chat agent for answering dashboard column / group questions.

Architecture
------------
- Uses a **tool-calling** ReAct agent so the LLM can look up specific columns
  or groups on demand rather than stuffing everything into the prompt.
- Metadata is loaded once and cached in-process; subsequent calls are fast.
- Stateless per-request: conversation history is passed in by the caller
  (API or Dash callback), making the agent compatible with any frontend.

Provider selection
------------------
Set ``CHAT_PROVIDER`` to choose the LLM backend:

  ``openai``  (default) — requires ``OPENAI_API_KEY``.
  ``gemini``            — requires ``GOOGLE_API_KEY``.

Model override: set ``CHAT_MODEL`` to any model name supported by the chosen
provider (e.g. ``gpt-4o-mini``, ``gemini-2.5-flash``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

# Load .env so API keys are available regardless of which entry point started the process.
_repo_root = Path(__file__).resolve().parent.parent.parent
for _env_candidate in [Path.cwd() / ".env", _repo_root / ".env"]:
    if _env_candidate.exists():
        load_dotenv(dotenv_path=str(_env_candidate), override=False)
        break

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from dashboards.metadata_builder import build_metadata, format_metadata_context

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Module-level singleton — built on first call, reused for process lifetime
# ---------------------------------------------------------------------------
_METADATA: Optional[Dict[str, Any]] = None
_METADATA_TEXT: Optional[str] = None


def _ensure_metadata(
    dashboard_config_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    global _METADATA, _METADATA_TEXT
    if _METADATA is None or force_rebuild:
        _METADATA = build_metadata(
            dashboard_config_path=dashboard_config_path,
            force_rebuild=force_rebuild,
        )
        _METADATA_TEXT = format_metadata_context(_METADATA)
    return _METADATA


def _get_metadata_text() -> str:
    if _METADATA_TEXT is None:
        _ensure_metadata()
    return _METADATA_TEXT or ""


# ---------------------------------------------------------------------------
# LangChain Tools — callable by the agent during reasoning
# ---------------------------------------------------------------------------


@tool
def lookup_column(column_name: str) -> str:
    """Look up the definition, formula, and category of a dashboard column by name.

    Use this when a user asks about a specific metric or column.
    Supports fuzzy matching — partial names will return close matches.
    """
    meta = _ensure_metadata()
    columns = meta.get("columns", {})

    if column_name in columns:
        info = columns[column_name]
        parts = [f"**{column_name}**"]
        parts.append(f"Category: {info.get('category', 'unknown')}")
        parts.append(f"Description: {info.get('description', 'No description.')}")
        if info.get("formula"):
            parts.append(f"Formula: {info['formula']}")
        etl = info.get("etl_formula", {})
        if isinstance(etl, dict):
            for src, expr in etl.items():
                parts.append(f"ETL source ({src}): `{expr}`")
        return "\n".join(parts)

    # Fuzzy match
    query_lower = column_name.lower().replace(" ", "_")
    matches = [c for c in columns if query_lower in c.lower() or c.lower() in query_lower]
    if matches:
        results = [f"No exact match for '{column_name}'. Close matches:"]
        for m in matches[:10]:
            desc = columns[m].get("description", "")
            results.append(f"- **{m}**: {desc[:120]}")
        return "\n".join(results)

    return f"Column '{column_name}' not found in metadata. Available categories: dimension, user_metric, aggregate_metric, retention, fish_hunter."


@tool
def lookup_group(group_name: str) -> str:
    """Look up how a user group or segment is defined.

    Use this when a user asks about group definitions like 'new', 'old',
    'beginner', 'AI', 'Default', 'day0_user', etc.
    """
    meta = _ensure_metadata()
    groups = meta.get("groups", {})

    for gname, ginfo in groups.items():
        values = ginfo.get("values", {})
        if group_name.lower() in (v.lower() for v in values):
            val_info = values.get(group_name) or values.get(group_name.lower(), {})
            if not val_info:
                for k, v in values.items():
                    if k.lower() == group_name.lower():
                        val_info = v
                        break
            parts = [
                f"**{group_name}** (part of `{gname}`)",
                f"Parent group description: {ginfo.get('description', '')}",
                f"Definition: {val_info.get('definition', 'N/A')}",
                f"Description: {val_info.get('description', 'N/A')}",
            ]
            return "\n".join(parts)

        if group_name.lower() == gname.lower():
            parts = [f"**{gname}**", f"Description: {ginfo.get('description', '')}"]
            for vname, vinfo in values.items():
                parts.append(
                    f"- **{vname}**: {vinfo.get('description', '')} " f"(SQL: `{vinfo.get('definition', 'N/A')}`)"
                )
            return "\n".join(parts)

    return f"Group '{group_name}' not found. Known groups: {', '.join(groups.keys())}."


@tool
def list_columns_by_category(category: str) -> str:
    """List all columns in a given category.

    Categories: dimension, user_metric, aggregate_metric, retention, fish_hunter, etl_derived.
    """
    meta = _ensure_metadata()
    columns = meta.get("columns", {})
    matches = [
        (name, info.get("description", "")[:100])
        for name, info in columns.items()
        if info.get("category", "") == category
    ]
    if not matches:
        all_cats = sorted({info.get("category", "other") for info in columns.values()})
        return f"No columns in category '{category}'. Available: {', '.join(all_cats)}"

    lines = [f"Columns in category '{category}' ({len(matches)} total):\n"]
    for name, desc in matches:
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)


@tool
def get_dashboard_config_summary() -> str:
    """Get a summary of the currently loaded dashboard configuration.

    Shows which game is active, what columns are displayed in each chart group,
    and the grouping/date columns.
    """
    meta = _ensure_metadata()
    config = meta.get("dashboard_config")
    if not config:
        return "No dashboard config is currently loaded."

    parts = [f"Dashboard: {config.get('title', 'Unknown')}"]
    sbd = config.get("stats_by_date", {})
    if sbd:
        parts.append(f"Group column: {sbd.get('group_col')}")
        parts.append(f"Date column: {sbd.get('date_col')}")
        parts.append(f"User group columns: {sbd.get('user_group_cols')}")
        for i in range(1, 10):
            label = sbd.get(f"group{i}_label")
            cols = sbd.get(f"group{i}_columns")
            if label and cols:
                parts.append(f"\n**{label}** ({len(cols)} metrics):")
                for c in cols:
                    parts.append(f"  - {c}")

    wr = config.get("weekly_report", {})
    if wr:
        parts.append(f"\nWeekly report game: {wr.get('game_id')}")

    return "\n".join(parts)


@tool
def get_game_info(game_id: str) -> str:
    """Get information about a specific game (SS01, SS02, SS03, FM01)."""
    meta = _ensure_metadata()
    games = meta.get("games", {})
    game = games.get(game_id.upper())
    if not game:
        return f"Game '{game_id}' not found. Known games: {', '.join(games.keys())}"
    parts = [
        f"**{game_id.upper()}** — {game.get('full_name', '')}",
        f"Type: {game.get('type', '')}",
        f"Source table: {game.get('source_table', '')}",
    ]
    if game.get("bet_types"):
        parts.append(f"Bet types: {', '.join(game['bet_types'])}")
    if game.get("notes"):
        parts.append(f"Notes: {game['notes']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a data analytics assistant for a game analytics dashboard.
You help users understand dashboard metrics, column definitions, user group
segmentations, and game-specific terminology.

Your knowledge comes from:
1. A curated column metadata file covering all dashboard columns
2. ETL source code that computes these columns from Redshift SQL
3. The dashboard configuration YAML for the currently active game

Guidelines:
- When asked about a column, use the lookup_column tool to get its exact definition.
- When asked about user groups (new/old/beginner/AI/Default), use lookup_group.
- When asked what columns are available, use list_columns_by_category.
- For game-specific questions, use get_game_info.
- Be precise about formulas and SQL definitions.
- If a column has both a hand-curated description and ETL-derived formula, include both.
- Explain RTP (Return to Player) as total_payout / total_bet when relevant.
- Explain the difference between user-level metrics (per-player) and
  group-level aggregates when the user seems confused.
- Answer in the same language the user uses (English or Chinese).
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_TOOLS = [
    lookup_column,
    lookup_group,
    list_columns_by_category,
    get_dashboard_config_summary,
    get_game_info,
]


_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


def _build_llm() -> BaseChatModel:
    provider = os.environ.get("CHAT_PROVIDER", "openai").lower()
    model = os.environ.get("CHAT_MODEL", _DEFAULT_MODELS.get(provider, "gpt-4o-mini"))

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "CHAT_PROVIDER=gemini requires langchain-google-genai. "
                "Install with:  poetry add langchain-google-genai"
            )
        from pydantic import SecretStr

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini requires GOOGLE_API_KEY or GEMINI_API_KEY. " "Add to .env or export in your shell."
            )
        return ChatGoogleGenerativeAI(model=model, temperature=0, api_key=SecretStr(api_key))

    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI requires OPENAI_API_KEY. Add to .env or export in your shell.")
    return ChatOpenAI(model=model, temperature=0, api_key=SecretStr(api_key))


def create_agent():
    """Create and return the LangGraph ReAct agent."""
    llm = _build_llm()
    return create_react_agent(llm, _TOOLS)


def init_metadata(
    dashboard_config_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> None:
    """Pre-warm the metadata cache (call at app startup)."""
    _ensure_metadata(
        dashboard_config_path=dashboard_config_path,
        force_rebuild=force_rebuild,
    )


def chat(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    dashboard_config_path: Optional[Path] = None,
) -> str:
    """
    Send a message to the chat agent and return the response text.

    Parameters
    ----------
    user_message : str
        The user's question.
    history : list of dict, optional
        Previous messages as ``[{"role": "user"|"assistant", "content": "..."}, ...]``.
    dashboard_config_path : Path, optional
        Path to the active dashboard config YAML. Used to contextualize
        answers with the currently displayed game.

    Returns
    -------
    str
        The assistant's response.
    """
    _ensure_metadata(dashboard_config_path=dashboard_config_path)
    agent = create_agent()

    messages: List[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]

    for msg in history or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    result = agent.invoke({"messages": messages})

    response_messages = result.get("messages", [])
    for m in reversed(response_messages):
        if isinstance(m, AIMessage) and m.content and isinstance(m.content, str):
            return m.content

    return "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
