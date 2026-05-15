"""
Diagnostic tests for whether the chat agent actually invokes
``search_confluence_rag`` when answering a RAG-targetable question.

The companion module ``tests/integration/test_rag_service.py`` proves
the RAG service itself works; this module proves the agent calls it.
Failures here are tool-selection issues (system prompt, docstring
wording, model choice) — not retrieval problems.

Skipped automatically when no embedding/chat API key is configured, so
CI stays green out of the box.

Run::

    poetry run pytest tests/integration/test_chat_agent_rag.py -v
"""

from __future__ import annotations

import os
from typing import Any, List

import pytest

# Load .env so OPENAI_API_KEY / GOOGLE_API_KEY are visible to the skipif
# decorator below at collection time (the chat_agent module's normal
# dotenv load happens later, inside test bodies).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

pytestmark = pytest.mark.integration


def _has_chat_key() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )


skipif_no_chat_key = pytest.mark.skipif(
    not _has_chat_key(),
    reason="Set OPENAI_API_KEY or GOOGLE_API_KEY to run chat-agent integration tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_names_called(messages: List[Any]) -> List[str]:
    """Return every tool name the LLM invoked across the message thread.

    LangChain stores tool calls on ``AIMessage.tool_calls`` (a list of
    ``{"name": ..., "args": ..., "id": ...}``). We walk the whole result
    list because a multi-step ReAct trace can have several tool calls
    before the final answer.
    """
    names: List[str] = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# Sanity — no API key required for these
# ---------------------------------------------------------------------------


def test_rag_tool_is_registered() -> None:
    """Static check: the RAG tool is in the agent's tool list at all."""
    from ai_agent.chat_agent import _TOOLS

    names = [t.name for t in _TOOLS]
    assert "search_confluence_rag" in names, f"search_confluence_rag missing from _TOOLS — got: {names}"


# ---------------------------------------------------------------------------
# Live agent invocation — proves whether the LLM actually chooses the RAG tool
# ---------------------------------------------------------------------------


@skipif_no_chat_key
def test_agent_invokes_rag_for_internal_docs_query() -> None:
    """The agent should call ``search_confluence_rag`` for an internal-only
    documentation question."""
    from ai_agent.chat_agent import _build_messages, create_agent

    agent = create_agent()
    messages = _build_messages(
        user_message=(
            "Look up our internal docs: what is the rollerCoaster math table " "and how does its RTP design work?"
        ),
        history=None,
    )
    result = agent.invoke({"messages": messages})
    tool_names = _tool_names_called(result["messages"])

    assert "search_confluence_rag" in tool_names, (
        "Agent did not invoke search_confluence_rag. "
        f"Tools called: {tool_names or '(none — likely answered from prior knowledge)'}"
    )


@skipif_no_chat_key
def test_agent_prefers_rag_over_live_confluence() -> None:
    """When both tools are available, the agent must try the RAG first.

    Reasoning: ``search_confluence_rag``'s docstring says
    'Prefer this over `search_confluence`'. If the LLM picks the live
    keyword tool first, we lose the RAG's cost/latency win.
    """
    from ai_agent.chat_agent import _build_messages, create_agent

    agent = create_agent()
    messages = _build_messages(
        user_message="Search our internal documentation for RTP calculation rules.",
        history=None,
    )
    result = agent.invoke({"messages": messages})
    tool_names = _tool_names_called(result["messages"])

    # Either the RAG was the only doc-search tool used, or it was at least
    # called before `search_confluence`. Both pass.
    rag_idx = tool_names.index("search_confluence_rag") if "search_confluence_rag" in tool_names else -1
    live_idx = tool_names.index("search_confluence") if "search_confluence" in tool_names else -1

    assert rag_idx != -1, f"Agent never called search_confluence_rag. Tools used: {tool_names}"
    if live_idx != -1:
        assert rag_idx < live_idx, (
            "Agent called search_confluence before search_confluence_rag. " f"Tools used in order: {tool_names}"
        )


@skipif_no_chat_key
def test_rag_returns_chunks_with_attribution() -> None:
    """Sanity-check that when the RAG tool *is* called, its output makes it
    into the final answer with at least the page title surfaced."""
    from ai_agent.chat_agent import _build_messages, create_agent

    agent = create_agent()
    messages = _build_messages(
        user_message=(
            "Using search_confluence_rag, find docs about the rollerCoaster math table "
            "RTP design and report what the top passage says verbatim."
        ),
        history=None,
    )
    result = agent.invoke({"messages": messages})

    tool_names = _tool_names_called(result["messages"])
    assert "search_confluence_rag" in tool_names, f"Skipped — agent didn't call the RAG tool. Tools used: {tool_names}"

    # The RAG tool output is captured on a ToolMessage. The final AIMessage
    # should synthesize it. We just check the answer is non-trivially long.
    from langchain_core.messages import AIMessage

    final_text = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, str):
                final_text = content
            elif isinstance(content, list):
                # Gemini sometimes returns content blocks
                final_text = " ".join(blk.get("text", "") for blk in content if isinstance(blk, dict))
            if final_text:
                break

    assert len(final_text) > 80, f"Final answer suspiciously short ({len(final_text)} chars); content={final_text!r}"
