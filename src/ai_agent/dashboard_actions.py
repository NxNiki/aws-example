"""Dashboard-driving tools for the agent-native chat (Phase 3).

These LangChain tools let the chat agent OPERATE the React dashboard: when the
model calls one, the SSE endpoint (`/api/agent/chat`) intercepts the tool call
and forwards it to the browser as a structured ``action`` event, where the
frontend's action dispatcher validates it and mutates the UI store. The tool
bodies are deliberately no-ops that return a short confirmation string — the
real effect happens client-side; the string just lets the ReAct loop continue.

``ask_user`` is the clarify mechanism: the frontend renders the options as
quick-reply chips and the turn ends until the user answers.
"""

from __future__ import annotations

from typing import List

from langchain_core.tools import tool


@tool
def load_config(config_id: str) -> str:
    """Switch the dashboard to a different game config.

    API tool: emitted to the dashboard as a ``load_config`` action.
    ``config_id`` must be one of the available config ids listed in the
    DASHBOARD STATE context (e.g. ``fishhunter``, ``ss01``, ``ss02``).
    """
    return f"Dashboard action dispatched: switched to config '{config_id}'."


@tool
def navigate_tab(tab: str) -> str:
    """Switch the dashboard to another tab.

    API tool: emitted as a ``navigate_tab`` action. ``tab`` must be one of:
    ``stats-by-date`` (metric time series), ``stats-by-group`` (distribution
    comparison across cohorts/date ranges), ``stats-deepdive`` (histogram /
    correlation / scatter).
    """
    return f"Dashboard action dispatched: navigated to tab '{tab}'."


@tool
def set_metrics(panel_id: str, left_metrics: List[str], right_metrics: List[str]) -> str:
    """Set which metrics a Stats-by-Date panel plots on its left/right y-axes.

    API tool: emitted as a ``set_metrics`` action. ``panel_id`` is one of the
    panel group ids from the DASHBOARD STATE context (e.g. ``group1``); pick the
    panel whose available metrics contain the requested ones. Metrics not in
    that panel's list are ignored by the dashboard.
    """
    return f"Dashboard action dispatched: panel '{panel_id}' now plots " f"left={left_metrics} right={right_metrics}."


@tool
def set_date_range(date_from: str, date_to: str) -> str:
    """Set the Stats-by-Date tab's date window.

    API tool: emitted as a ``set_date_range`` action. Dates are inclusive ISO
    strings (YYYY-MM-DD) and must lie within the data bounds shown in the
    DASHBOARD STATE context.
    """
    return f"Dashboard action dispatched: date range set to {date_from} → {date_to}."


@tool
def set_granularity(granularity: str) -> str:
    """Set the Stats-by-Date tab's time granularity.

    API tool: emitted as a ``set_granularity`` action. One of ``day``,
    ``week``, ``month`` (only those listed for the active config).
    """
    return f"Dashboard action dispatched: granularity set to '{granularity}'."


@tool
def ask_user(question: str, options: List[str]) -> str:
    """Ask the user a clarifying question with quick-reply options.

    API tool: emitted as a ``clarify`` event; the dashboard shows ``options``
    as one-click reply chips. Call this when a request is ambiguous (e.g.
    "show active users" — by date or by group?) instead of guessing. After
    calling it, STOP — do not call more tools or continue answering; the user's
    reply arrives as the next message.
    """
    return "Question shown to the user with quick-reply options. END your turn now and wait for the reply."


ACTION_TOOLS = [load_config, navigate_tab, set_metrics, set_date_range, set_granularity, ask_user]
ACTION_TOOL_NAMES = {t.name for t in ACTION_TOOLS}
CLARIFY_TOOL_NAME = ask_user.name
