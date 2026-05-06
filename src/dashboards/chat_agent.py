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

You can also pass a catalog key like ``"openai:gpt-4.1"`` or
``"gemini:gemini-2.5-pro"`` via the ``model_key`` parameter in
``chat()`` / ``achat()`` to select both provider and model at once.
"""

from __future__ import annotations

import json
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


def _format_column(name: str, info: Dict[str, Any]) -> str:
    parts = [f"**{name}**"]
    parts.append(f"Category: {info.get('category', 'unknown')}")
    parts.append(f"Description: {info.get('description', 'No description.')}")
    if info.get("formula"):
        parts.append(f"Formula: {info['formula']}")
    etl = info.get("etl_formula", {})
    if isinstance(etl, dict):
        for src, expr in etl.items():
            parts.append(f"ETL source ({src}): `{expr}`")
    if info.get("python_aggregation"):
        parts.append(f"Python aggregation code:\n```python\n{info['python_aggregation']}\n```")
    return "\n".join(parts)


@tool
def lookup_column(column_name: str) -> str:
    """Look up the definition, formula, and category of a dashboard column by name.

    Use this when a user asks about a specific metric or column.
    Supports fuzzy matching — partial names will return close matches.
    """
    meta = _ensure_metadata()
    columns = meta.get("columns", {})

    if column_name in columns:
        return _format_column(column_name, columns[column_name])

    query_lower = column_name.lower().replace(" ", "_")
    matches = [c for c in columns if query_lower in c.lower() or c.lower() in query_lower]
    if matches:
        results = [f"No exact match for '{column_name}'. Close matches:"]
        for m in matches[:10]:
            desc = columns[m].get("description", "")
            results.append(f"- **{m}**: {desc[:120]}")
        return "\n".join(results)

    return (
        f"Column '{column_name}' not found by name. "
        "Try search_columns_by_keyword with descriptive keywords, "
        "or list_columns_by_category to browse available columns."
    )


@tool
def search_columns_by_keyword(keyword: str) -> str:
    """Search columns by keyword across both names AND descriptions.

    Use this when the user describes a metric conceptually (e.g. "users who
    lose a lot", "retention rate", "how many people came back") rather than
    by exact column name.

    Args:
        keyword: One or two descriptive words (e.g. "rtp", "retention", "loss", "payout", "bet count")
    """
    meta = _ensure_metadata()
    columns = meta.get("columns", {})
    kw = keyword.lower()

    matches = []
    for name, info in columns.items():
        desc = info.get("description", "").lower()
        formula = info.get("formula", "").lower()
        if kw in name.lower() or kw in desc or kw in formula:
            matches.append((name, info.get("description", "")[:120]))

    if not matches:
        return f"No columns match '{keyword}'. " "Try broader keywords or use list_columns_by_category to browse."

    lines = [f"Found {len(matches)} column(s) matching '{keyword}':\n"]
    for name, desc in matches[:15]:
        lines.append(f"- **{name}**: {desc}")
    if len(matches) > 15:
        lines.append(f"\n... and {len(matches) - 15} more.")
    return "\n".join(lines)


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
# ETL source reading tool
# ---------------------------------------------------------------------------


@tool
def read_etl_source(file_path: str, column_name: str = "") -> str:
    """Read the full SQL query from an ETL source file.

    Use this when a user asks for the actual SQL code or detailed computation
    logic behind a column. The ETL sources are scanned at startup and their
    SQL snippets are stored in the metadata cache.

    Args:
        file_path: Path to the ETL file (e.g. "jobs/fish_hunter/etl_game_stats_daily_by_user.py").
                   Use get_dashboard_config_summary or lookup_column to discover which ETL files
                   are relevant for the current dashboard.
        column_name: Optional column name to highlight in the SQL output.
    """
    meta = _ensure_metadata()
    etl_sources = meta.get("etl_sources", [])

    matching = [s for s in etl_sources if file_path in s.get("file", "")]
    if not matching:
        available = [s.get("file", "?") for s in etl_sources]
        return f"ETL file '{file_path}' not found in scanned sources. Available:\n" + "\n".join(
            f"- {f}" for f in available
        )

    source = matching[0]
    snippets = source.get("sql_snippets", [])
    if not snippets:
        return f"No SQL snippets found in '{source['file']}'."

    parts = [f"ETL file: **{source['file']}**"]
    if source.get("game_id"):
        parts.append(f"Game ID: {source['game_id']}")
    parts.append(f"SQL snippets found: {len(snippets)}\n")

    for i, snippet in enumerate(snippets, 1):
        if column_name:
            if column_name.lower() not in snippet.lower():
                continue
        header = f"--- SQL Snippet {i} ---"
        # Truncate very long snippets
        if len(snippet) > 4000:
            snippet = snippet[:4000] + "\n-- ... (truncated, query too long)"
        parts.append(f"{header}\n```sql\n{snippet}\n```\n")

    if len(parts) <= 3 and column_name:
        parts.append(f"Column '{column_name}' not found in SQL snippets. Showing all snippets:")
        for i, snippet in enumerate(snippets, 1):
            if len(snippet) > 4000:
                snippet = snippet[:4000] + "\n-- ... (truncated)"
            parts.append(f"--- SQL Snippet {i} ---\n```sql\n{snippet}\n```\n")

    # Also include Python aggregation logic if available
    agg_constants = meta.get("python_aggregation_constants", "")
    if agg_constants:
        parts.append(f"Dashboard aggregation constants:\n```python\n{agg_constants}\n```")

    return "\n".join(parts)


@tool
def list_etl_sources() -> str:
    """List all ETL source files that have been scanned for column computation logic.

    Use this to discover which ETL files are available, then use read_etl_source
    to read the actual SQL from a specific file.
    """
    meta = _ensure_metadata()
    etl_sources = meta.get("etl_sources", [])
    if not etl_sources:
        return "No ETL source files have been scanned."

    lines = [f"Scanned {len(etl_sources)} ETL source file(s):\n"]
    for s in etl_sources:
        game = s.get("game_id") or "all"
        aliases = len(s.get("sql_column_aliases", {}))
        snippets = s.get("sql_snippets_count", 0)
        lines.append(f"- **{s['file']}** (game: {game}, {aliases} aliases, {snippets} SQL snippets)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confluence tools
# ---------------------------------------------------------------------------


@tool
def search_confluence(query: str, space_key: str = "") -> str:
    """Search Confluence documentation for pages matching a query.

    Use this when the user asks about processes, policies, documentation,
    or anything not covered by the dashboard metadata tools.

    Args:
        query: Search keywords (e.g. "RTP calculation", "user segmentation rules")
        space_key: Optional Confluence space key to narrow the search
    """
    try:
        from dashboards.confluence_client import search_pages as _search
    except ImportError:
        return "Confluence integration is not available (atlassian-python-api not installed)."
    try:
        pages = _search(query, space_key=space_key or None, max_results=5)
    except Exception as exc:
        return f"Confluence search failed: {exc}"

    if not pages:
        return f"No Confluence pages found for '{query}'."

    lines = [f"Found {len(pages)} page(s) for '{query}':\n"]
    for p in pages:
        lines.append(f"- **{p['title']}** (id: {p['id']}, space: {p['space']})")
        if p.get("excerpt"):
            lines.append(f"  _{p['excerpt']}_")
    lines.append("\nUse `read_confluence_page` with the page id to read the full content.")
    return "\n".join(lines)


@tool
def read_confluence_page(page_id: str) -> str:
    """Read the full content of a Confluence page by its ID.

    Use this after search_confluence returns page IDs to read the actual content
    and answer the user's question based on the documentation.

    Args:
        page_id: The Confluence page ID (numeric string from search results)
    """
    try:
        from dashboards.confluence_client import get_page_content as _get_page
    except ImportError:
        return "Confluence integration is not available (atlassian-python-api not installed)."
    try:
        return _get_page(page_id)
    except Exception as exc:
        return f"Failed to read Confluence page {page_id}: {exc}"


@tool
def search_confluence_rag(query: str, top_k: int = 5) -> str:
    """Semantically search the curated Confluence RAG index.

    Prefer this over `search_confluence` for any documentation question — it
    returns the most relevant passages (BM25 + vector hybrid) instead of just
    keyword hits, and is much faster than live Confluence calls.

    Falls back automatically with an explanatory message if the RAG service
    is unreachable; in that case use `search_confluence` followed by
    `read_confluence_page` as a backup.

    Args:
        query: A natural-language question or descriptive phrase.
        top_k: Number of passages to return (default 5).
    """
    try:
        from rag_service.client import retrieve_passages
    except ImportError:
        return "RAG service client is not installed in this environment."
    return retrieve_passages(query, top_k=top_k)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a data analytics assistant for a game analytics dashboard.
You help users understand dashboard metrics, column definitions, user group
segmentations, game-specific terminology, and team documentation.

Your knowledge comes from:
1. A curated column metadata file covering all dashboard columns
2. ETL source code (Redshift SQL) that computes per-user statistics
3. Python aggregation code (user_stats_aggregates.py) that derives dashboard
   metrics from ETL output — e.g. filtering users, computing ratios
4. The dashboard configuration YAML for the currently active game
5. Confluence documentation (searchable and readable via tools)

IMPORTANT — how to answer column questions:
- ALWAYS use lookup_column first. It returns the full description, formula,
  ETL SQL expression, AND Python aggregation code for the column.
- Include ALL details from the tool output in your answer: the description,
  the threshold/filter conditions, the formula, and the code. NEVER
  simplify or omit conditions (like minimum bet thresholds).
- If the user asks for the actual SQL or code, use read_etl_source with the
  ETL file path shown in lookup_column's output. Use list_etl_sources to
  discover available ETL files if needed.
- If the user describes a metric conceptually (e.g. "users who lose a lot",
  "how many people came back"), use search_columns_by_keyword with
  descriptive keywords extracted from their question.
- If search_columns_by_keyword returns no results, try different keywords or
  use list_columns_by_category to browse and find the closest match.
- NEVER repeat a previous answer if you cannot find the column. Instead, tell
  the user what you searched for and suggest they clarify.

Understanding the data pipeline:
- ETL SQL queries run on Redshift and produce per-user per-day rows
  (columns prefixed with user_*: user_num_bets, user_total_bet, user_rtp, etc.)
- The dashboard Python code (DataMetrics class) reads these user-level rows
  and computes group-level aggregate metrics (num_active_users, rtp, hit_rate,
  etc.) using thresholds like ACTIVE_USER_MIN_BETS = 40.
- When explaining aggregate metrics, always mention both the ETL source
  (how per-user data is computed) and the Python aggregation (how users are
  filtered and aggregated into the dashboard metric).

Other guidelines:
- When asked about user groups (new/old/beginner/AI/Default), use lookup_group.
- For game-specific questions, use get_game_info.
- For questions about processes, policies, or documentation, FIRST try
  search_confluence_rag (the curated RAG index — fast, semantic). If it
  returns nothing useful or reports the service is unavailable, fall back
  to search_confluence + read_confluence_page (live Confluence API).
- Be precise about formulas and SQL definitions.
- If a column has both a hand-curated description and ETL-derived formula, include both.
- Explain RTP (Return to Player) as total_payout / total_bet when relevant.
- When citing Confluence pages, mention the page title so the user can find it.
- Answer in the same language the user uses (English or Chinese).
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_TOOLS = [
    lookup_column,
    search_columns_by_keyword,
    lookup_group,
    list_columns_by_category,
    get_dashboard_config_summary,
    get_game_info,
    read_etl_source,
    list_etl_sources,
    search_confluence_rag,
    search_confluence,
    read_confluence_page,
]


_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}

# Full model catalog — keyed by "provider:model" for the UI dropdown.
# "tier" is just informational for the dropdown label.
MODEL_CATALOG = {
    "openai:gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini", "tier": "fast"},
    "openai:gpt-4o": {"provider": "openai", "model": "gpt-4o", "tier": "balanced"},
    "openai:gpt-4.1-mini": {"provider": "openai", "model": "gpt-4.1-mini", "tier": "balanced"},
    "openai:gpt-4.1": {"provider": "openai", "model": "gpt-4.1", "tier": "powerful"},
    "gemini:gemini-2.5-flash": {"provider": "gemini", "model": "gemini-2.5-flash", "tier": "fast"},
    "gemini:gemini-2.5-pro": {"provider": "gemini", "model": "gemini-2.5-pro", "tier": "powerful"},
}

# Secrets Manager secret name (stores GOOGLE_API_KEY and OPENAI_API_KEY).
_SECRETS_MANAGER_NAME = "ai-dashboard_ai_agent"
_SECRETS_MANAGER_REGION = "us-west-2"
_secrets_cache: Optional[Dict[str, str]] = None


def _get_secret(key: str) -> Optional[str]:
    """Return an API key from env var first, then AWS Secrets Manager."""
    value = os.environ.get(key)
    if value:
        return value

    global _secrets_cache
    if _secrets_cache is None:
        try:
            import json as _json

            import boto3  # type: ignore[import-untyped]

            session = boto3.Session(region_name=_SECRETS_MANAGER_REGION)
            client = session.client(service_name="secretsmanager")
            resp = client.get_secret_value(SecretId=_SECRETS_MANAGER_NAME)
            _secrets_cache = _json.loads(resp["SecretString"])
        except Exception as exc:
            logger.warning("Secrets Manager lookup failed: %s", exc)
            _secrets_cache = {}

    return (_secrets_cache or {}).get(key)


def _build_llm(model_key: Optional[str] = None) -> BaseChatModel:
    """Build the LLM instance.

    Args:
        model_key: A catalog key like ``"openai:gpt-4.1"`` or ``"gemini:gemini-2.5-pro"``.
                   Falls back to ``CHAT_PROVIDER`` / ``CHAT_MODEL`` env vars.

    ``CHAT_PROVIDER`` defaults to ``"auto"``: prefer OpenAI if its key is set,
    otherwise fall back to Gemini if its key is set. Mirrors the
    rag_service embedder's auto-mode so a single ``GOOGLE_API_KEY`` (or
    ``OPENAI_API_KEY``) is enough to bring up both surfaces.
    """
    if model_key and model_key in MODEL_CATALOG:
        entry = MODEL_CATALOG[model_key]
        provider = entry["provider"]
        model = entry["model"]
    else:
        provider = os.environ.get("CHAT_PROVIDER", "auto").lower()
        if provider == "auto":
            if _get_secret("OPENAI_API_KEY"):
                provider = "openai"
            elif _get_secret("GOOGLE_API_KEY") or _get_secret("GEMINI_API_KEY"):
                provider = "gemini"
            else:
                raise ValueError(
                    "No LLM credentials found. Set OPENAI_API_KEY or "
                    "GOOGLE_API_KEY/GEMINI_API_KEY in .env, your shell, or "
                    "Secrets Manager. Or pin CHAT_PROVIDER to a specific provider."
                )
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

        api_key = _get_secret("GOOGLE_API_KEY") or _get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini requires GOOGLE_API_KEY or GEMINI_API_KEY. "
                "Add to .env, export in your shell, or store in Secrets Manager."
            )
        return ChatGoogleGenerativeAI(model=model, temperature=0, api_key=SecretStr(api_key))

    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI requires OPENAI_API_KEY. " "Add to .env, export in your shell, or store in Secrets Manager."
        )
    return ChatOpenAI(model=model, temperature=0, api_key=SecretStr(api_key))


def create_agent(model_key: Optional[str] = None):
    """Create and return the LangGraph ReAct agent."""
    llm = _build_llm(model_key=model_key)
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


def _build_messages(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[BaseMessage]:
    messages: List[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]
    for msg in history or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))
    return messages


def _stringify_ai_content(content: Any) -> str:
    """Turn ``AIMessage.content`` into plain text.

    Gemini and some LangChain providers return ``content`` as a list of blocks
    (e.g. ``[{"type": "text", "text": "..."}]``), which made the old
    ``isinstance(..., str)`` check fail and drop the real answer.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if t is not None:
                    parts.append(str(t))
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        if isinstance(content.get("content"), str):
            return str(content["content"]).strip()
    return str(content).strip()


_DEBUG_PREVIEW_MAX = 6000


def _format_debug_payload(content: Any) -> str:
    """Human-readable dump for in-chat debugging."""
    try:
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2, default=str)
    except Exception:
        pass
    return repr(content)


def _truncate_debug(text: str, max_len: int = _DEBUG_PREVIEW_MAX) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 30] + "\n\n… (truncated for chat)"


def _messages_for_current_turn(msgs: List[BaseMessage]) -> List[BaseMessage]:
    """Keep only messages after the last ``HumanMessage`` (the active user question).

    The graph appends multiple ``AIMessage`` / tool steps per turn. If the final
    ``AIMessage`` has no extractable text, scanning the **whole** thread walks
    backward into the **previous** user question's reply — producing the same
    answer for every new question.
    """
    last_human = -1
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            last_human = i
            break
    if last_human < 0:
        return msgs
    return msgs[last_human + 1 :]


def _extract_response(result: dict) -> str:
    msgs: List[BaseMessage] = list(result.get("messages", []))
    turn_msgs = _messages_for_current_turn(msgs)

    for m in reversed(turn_msgs):
        if isinstance(m, AIMessage):
            text = _stringify_ai_content(m.content)
            if text:
                return text
            # Model returned something we could not stringify — show it in the chat UI.
            payload = _format_debug_payload(m.content)
            logger.warning(
                "Agent AIMessage had no extractable text; content=%r",
                m.content,
            )
            return (
                "**Could not turn the model reply into plain text.** "
                "Raw `AIMessage.content`:\n\n```\n" + _truncate_debug(payload) + "\n```"
            )

    # No AIMessage in this turn
    tail_types = [type(x).__name__ for x in turn_msgs[-12:]]
    logger.warning(
        "Agent result had no AIMessage in current turn; turn_len=%d total=%d tail_types=%s",
        len(turn_msgs),
        len(msgs),
        tail_types,
    )
    return (
        "**Debug: no assistant message in the agent result.**\n\n"
        f"- Messages in current turn: {len(turn_msgs)}\n"
        f"- Total messages: {len(msgs)}\n"
        f"- Last message types this turn (up to 12): `{', '.join(tail_types) or 'none'}`\n\n"
        "If this persists, check CloudWatch logs for the full trace."
    )


def chat(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    dashboard_config_path: Optional[Path] = None,
    model_key: Optional[str] = None,
) -> str:
    """
    Synchronous chat — used by the Dash embedded panel (Gunicorn / threaded).
    """
    _ensure_metadata(dashboard_config_path=dashboard_config_path)
    agent = create_agent(model_key=model_key)
    messages = _build_messages(user_message, history)
    result = agent.invoke({"messages": messages})
    return _extract_response(result)


async def achat(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    dashboard_config_path: Optional[Path] = None,
    model_key: Optional[str] = None,
) -> str:
    """
    Async chat — used by the FastAPI standalone service (Uvicorn / ASGI).

    Uses ``ainvoke`` so the event loop is never blocked while waiting for
    the LLM API response, allowing Uvicorn to handle many concurrent
    requests on a single worker.
    """
    _ensure_metadata(dashboard_config_path=dashboard_config_path)
    agent = create_agent(model_key=model_key)
    messages = _build_messages(user_message, history)
    result = await agent.ainvoke({"messages": messages})
    return _extract_response(result)
