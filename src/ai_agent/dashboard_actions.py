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

import json
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


def _validate_json_arg(arg_name: str, raw: str) -> str:
    """Return an ERROR string for the ReAct loop if ``raw`` isn't valid JSON.

    The action event is emitted to the browser at on_tool_start regardless, but
    the browser's toast is invisible to the model — this return value is the
    only way the agent learns its JSON was malformed and can self-correct.
    """
    try:
        json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"ERROR: {arg_name} is not valid JSON ({exc}). Fix the JSON and call the tool again."
    return ""


@tool
def set_report_period(date_from: str, date_to: str) -> str:
    """Set the Report tab's report period (the date window inherited figures use).

    API tool: emitted as a ``set_report_period`` action. Dates are inclusive ISO
    strings (YYYY-MM-DD). This only updates the spec — call ``regenerate_report``
    afterwards to re-fetch figures and refresh stale prose.
    """
    return f"Dashboard action dispatched: report period set to {date_from} → {date_to}."


@tool
def regenerate_report() -> str:
    """Re-fetch all report figures and refresh stale descriptions + summary.

    API tool: emitted as a ``regenerate_report`` action. Figures with
    ``inherit_period`` follow the report period; descriptions the user wrote by
    hand are never overwritten. Use after ``set_report_period`` or figure edits.
    """
    return "Dashboard action dispatched: report regeneration started."


@tool
def load_report_spec(name: str) -> str:
    """Load a saved Report Spec (and its linked dashboard view) into the Report tab.

    API tool: emitted as a ``load_report_spec`` action. ``name`` must be one of
    the saved spec names in the DASHBOARD STATE report section.
    """
    return f"Dashboard action dispatched: loading report spec '{name}'."


@tool
def save_report_spec(name: str) -> str:
    """Save the current Report Spec to S3 under ``name``.

    API tool: emitted as a ``save_report_spec`` action. Saving OVERWRITES any
    existing spec with the same name — only call this when the user explicitly
    asked to save, and confirm via ``ask_user`` if the name already exists.
    """
    return f"Dashboard action dispatched: saving report spec '{name}'."


@tool
def add_report_figure(title: str, source_json: str) -> str:
    """Add a figure to the report from a recipe.

    API tool: emitted as an ``add_report_figure`` action. ``source_json`` is a
    JSON object string — a complete figure source whose ``kind`` is one of
    ``stats-by-date``, ``stats-by-group``, ``stats-deepdive`` (full field shapes
    and examples are in the SKILLS section). Use config/panel/metric names from
    the DASHBOARD STATE only.
    """
    err = _validate_json_arg("source_json", source_json)
    return err or f"Dashboard action dispatched: figure '{title}' added to the report."


@tool
def patch_report_figure(figure_id: str, patch_json: str) -> str:
    """Edit one report figure: title, description, inherit_period, or source.

    API tool: emitted as a ``patch_report_figure`` action. ``patch_json`` is a
    JSON object string with only the fields to change; figure fields are
    shallow-merged but ``source``, if present, REPLACES the old source wholesale
    — always send a complete source object. Never copy truncated
    ``description_preview`` text from the DASHBOARD STATE into a patch.
    """
    err = _validate_json_arg("patch_json", patch_json)
    return err or f"Dashboard action dispatched: figure '{figure_id}' patched."


@tool
def remove_report_figure(figure_id: str) -> str:
    """Remove one figure from the report.

    API tool: emitted as a ``remove_report_figure`` action. ``figure_id`` must
    be a figure id from the DASHBOARD STATE report section.
    """
    return f"Dashboard action dispatched: figure '{figure_id}' removed."


ACTION_TOOLS = [
    load_config,
    navigate_tab,
    set_metrics,
    set_date_range,
    set_granularity,
    ask_user,
    set_report_period,
    regenerate_report,
    load_report_spec,
    save_report_spec,
    add_report_figure,
    patch_report_figure,
    remove_report_figure,
]
ACTION_TOOL_NAMES = {t.name for t in ACTION_TOOLS}
CLARIFY_TOOL_NAME = ask_user.name
