"""
Manual-review Q&A test harness for the chat agent.

Reads ``ai_agent_questions.yaml`` next to this file. Each entry is a
parametrized test case that:

1. Builds the chat agent (optionally pointed at a specific dashboard
   config + model).
2. Runs the question through ``agent.invoke``.
3. Prints the question, the answer, the tools the agent invoked, and
   the elapsed time at log-level INFO. Pytest's ``log_cli = true`` in
   ``tests/pytest.ini`` puts these in the terminal — no ``-s`` flag
   needed, though ``-s`` doesn't hurt.
4. Runs optional assertions:
     * ``must_contain`` — every listed substring must appear in the
       answer (case-insensitive).
     * ``must_call_tools`` — every listed tool name must appear in
       the agent's tool-call trace.
   If both lists are empty/missing, the test passes as long as the
   agent responds at all (smoke mode — review the printed output).

Run::

    poetry run pytest tests/integration/test_ai_agent_questions.py -v
    poetry run pytest tests/integration/test_ai_agent_questions.py -v -k ss03

Skipped automatically when no chat API key (OpenAI / Gemini) is
configured, so CI stays green without credentials.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

# Load .env so OPENAI_API_KEY / GOOGLE_API_KEY are visible to the
# skipif decorator at collection time.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.aws]

logger = logging.getLogger(__name__)


_QUESTIONS_FILE = Path(__file__).resolve().parent / "ai_agent_questions.yaml"


def _has_chat_key() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )


skipif_no_chat_key = pytest.mark.skipif(
    not _has_chat_key(),
    reason="Set OPENAI_API_KEY / GOOGLE_API_KEY / GEMINI_API_KEY to run AI-agent Q&A tests.",
)


def _load_cases() -> List[Dict[str, Any]]:
    """Read the YAML and return its ``cases`` list. Empty list if the
    file is missing so pytest collection doesn't error out — the user
    may not have authored questions yet."""
    if not _QUESTIONS_FILE.exists():
        return []
    with _QUESTIONS_FILE.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cases = raw.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError(f"{_QUESTIONS_FILE}: top-level 'cases' must be a list")
    return cases


def _tool_names_called(messages: List[Any]) -> List[str]:
    """Walk every message in the agent's result thread and collect
    each tool name the LLM invoked."""
    names: List[str] = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                names.append(name)
    return names


def _resolve_dashboard_config(entry: Dict[str, Any]) -> Any:
    """Convert the optional ``dashboard_config`` filename to an absolute
    path under ``configs/dashboard/``. Returns ``None`` when unset."""
    name = entry.get("dashboard_config")
    if not name:
        return None
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "dashboard" / name


_CASES = _load_cases()


@skipif_no_chat_key
@pytest.mark.parametrize("entry", _CASES, ids=[c.get("id", f"case_{i}") for i, c in enumerate(_CASES)])
def test_ai_agent_question(entry: Dict[str, Any]) -> None:
    """Run one Q&A entry through the agent and log the answer + tool calls.

    Assertions only fire for fields the entry actually populates, so
    an unannotated entry behaves as a manual-review smoke test.
    """
    # Imported lazily so module-level pytest collection works even when
    # the LangChain stack is not installed (e.g., dashboard-only dev env).
    from ai_agent.chat_agent import _build_messages, _ensure_metadata, _extract_response, create_agent

    _ensure_metadata(dashboard_config_path=_resolve_dashboard_config(entry))

    agent = create_agent(model_key=entry.get("model"))
    messages = _build_messages(user_message=entry["question"], history=None)

    t0 = time.monotonic()
    # invoke() wants langchain.agents._InputAgentState (private TypedDict);
    # plain dict is structurally equivalent at runtime.
    result = agent.invoke({"messages": messages})  # type: ignore[arg-type]
    elapsed_s = time.monotonic() - t0

    answer = _extract_response(result)
    tools_called = _tool_names_called(result.get("messages", []))

    # Loud, structured output for the manual reviewer. log_cli=true in
    # pytest.ini routes this to the terminal.
    logger.info("─" * 72)
    logger.info("[%s] %s", entry.get("id", "(no-id)"), entry["question"])
    if entry.get("dashboard_config"):
        logger.info("  config: %s", entry["dashboard_config"])
    if entry.get("model"):
        logger.info("  model:  %s", entry["model"])
    logger.info("  tools:  %s", tools_called or "(none called)")
    logger.info("  took:   %.1fs", elapsed_s)
    logger.info("  answer:")
    for line in (answer or "(empty)").splitlines() or ["(empty)"]:
        logger.info("    %s", line)
    logger.info("─" * 72)

    # Optional assertions — populated entries only.
    must_contain = entry.get("must_contain") or []
    answer_lower = (answer or "").lower()
    missing = [s for s in must_contain if s.lower() not in answer_lower]
    assert not missing, f"Answer missing required substrings: {missing}\nFull answer:\n{answer}"

    must_call_tools = entry.get("must_call_tools") or []
    not_called = [t for t in must_call_tools if t not in tools_called]
    assert not not_called, f"Agent did not call required tools: {not_called}. Called: {tools_called}"
