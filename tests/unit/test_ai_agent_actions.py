"""Unit tests for the agent's report action tools + skills registry (Phase 4b).

These need the ai_agent dependency set (langchain); CI lanes that install only
``dashboard_api`` skip them via importorskip.
"""

import json

import pytest

pytest.importorskip("langchain_core", reason="ai_agent llm deps not installed")

from langchain_core.messages import SystemMessage  # noqa: E402

from ai_agent.chat_api import _agent_context_message, _with_skills, create_chat_app, load_skills_markdown  # noqa: E402
from ai_agent.dashboard_actions import ACTION_TOOL_NAMES, add_report_figure, patch_report_figure  # noqa: E402

REPORT_TOOL_NAMES = {
    "set_report_period",
    "regenerate_report",
    "load_report_spec",
    "save_report_spec",
    "add_report_figure",
    "patch_report_figure",
    "remove_report_figure",
}


def test_report_action_tools_registered():
    assert REPORT_TOOL_NAMES <= ACTION_TOOL_NAMES


def test_json_string_args_validated_for_self_correction():
    # The browser toast is invisible to the model: a malformed-JSON arg must
    # come back as an ERROR tool result so the ReAct loop can retry.
    bad = add_report_figure.invoke({"title": "t", "source_json": "{not json"})
    assert bad.startswith("ERROR:")
    ok = add_report_figure.invoke({"title": "t", "source_json": json.dumps({"kind": "stats-by-date"})})
    assert not ok.startswith("ERROR:")
    assert patch_report_figure.invoke({"figure_id": "f1", "patch_json": "nope"}).startswith("ERROR:")


def test_skills_markdown_loads_and_names_the_skills():
    md = load_skills_markdown()
    for skill in ("## update_report", "## show_metric", "## edit_report_figure"):
        assert skill in md
    # the per-kind source examples the model copies from
    for kind in ("stats-by-date", "stats-by-group", "stats-deepdive"):
        assert kind in md


def test_agent_context_is_one_system_message_without_skills():
    # Gemini supports only ONE system instruction (merged by _merge_system) —
    # and the skills playbook must NOT be in it: gemini-2.5-flash returns an
    # empty response when instructional skill text rides in the system prompt.
    msg = _agent_context_message({"report": {"figures": []}, "configId": "ss01"})
    assert isinstance(msg, SystemMessage)
    assert "DASHBOARD STATE (current UI):" in msg.content
    assert '"configId": "ss01"' in msg.content
    assert "update_report" not in msg.content  # skills travel in the user turn


def test_skills_travel_in_the_user_turn():
    wrapped = _with_skills("update the report to May")
    assert wrapped.startswith("[CONTEXT — agent skills playbook")
    assert "## update_report" in wrapped
    assert wrapped.rstrip().endswith("User request: update the report to May")


def test_agent_context_truncates_oversized_state():
    msg = _agent_context_message({"panels": ["x" * 20000]})
    assert msg.content.endswith("…(truncated)")


def test_skills_route_exposed():
    app = create_chat_app()
    assert any(getattr(r, "path", "") == "/api/agent/skills" for r in app.routes)
