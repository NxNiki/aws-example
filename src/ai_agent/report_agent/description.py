"""
LLM helpers for the Report tab.

Two entry points:
  * ``generate_description`` — interpret a single figure given its
    ``data_summary`` (and optionally an existing description the user wants
    refined). Returns ``(text, status)``.
  * ``generate_summary`` — synthesize an overall report summary across all
    figures' descriptions + data summaries. Same return shape.

Generate semantics
------------------
The Generate buttons are intentionally *intent-driven*: clicking Generate
does not blindly overwrite manual edits. Behavior branches on the
textarea's current contents:

1. **Empty textarea** → LLM produces a first draft from the data summary.
   Status: ``STATUS_GENERATED``.
2. **Non-empty, no ``/prompt`` lines** → no LLM call; the textarea is
   returned unchanged. Status: ``STATUS_NO_INSTRUCTIONS``. The caller
   surfaces this as a status hint so the user knows nothing happened.
3. **Non-empty, ``/prompt`` line(s) at the very start (no preserved
   prose)** → full regenerate from scratch using the instructions.
   Status: ``STATUS_REGENERATED``.
4. **Non-empty, ``/prompt`` line(s) after some prose** → the prose
   *before* the first ``/prompt`` is preserved byte-for-byte and the
   LLM generates additional paragraphs based on the instructions; the
   two are concatenated. Status: ``STATUS_APPENDED``.

The same ``chat_agent._build_llm()`` factory is used so model selection +
credentials behave identically to the AI Assistant chat panel.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "zh-Hans": "Simplified Chinese",
    "zh-Hant": "Traditional Chinese",
}


def _language_name(language_code: str) -> str:
    return SUPPORTED_LANGUAGES.get(language_code, "English")


# Lines starting with ``/prompt`` (optional trailing colon) are user
# instructions, not prose to keep. One per line, case-insensitive. The
# trigger must be at the start of the line (after optional indentation) —
# mid-line occurrences of ``/prompt`` in prose are not stripped.
# Accepted forms (both produce identical parsing):
#   /prompt make this more concise
#   /prompt: make this more concise
_PROMPT_LINE_RE = re.compile(r"^[ \t]*/prompt(?::|\s)\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _split_at_first_prompt(text: Optional[str]) -> Tuple[str, List[str]]:
    """Split a textarea value at the first ``/prompt`` line.

    Returns ``(preserved_prose, prompts)``:

    * ``preserved_prose`` — everything before the first ``/prompt`` line,
      with trailing whitespace trimmed. This text is preserved verbatim
      when the caller appends new LLM content beneath it.
    * ``prompts`` — list of all ``/prompt`` instruction texts (from
      anywhere in the textarea, in document order).

    If no ``/prompt`` line is present, returns ``(text or "", [])``.
    ``None`` or empty returns ``("", [])``.
    """
    if not text:
        return "", []
    first = _PROMPT_LINE_RE.search(text)
    if not first:
        return text, []
    preserved = text[: first.start()].rstrip()
    after = text[first.start() :]
    prompts = [m.group(1).strip() for m in _PROMPT_LINE_RE.finditer(after)]
    return preserved, prompts


# Status strings returned by ``generate_description`` / ``generate_summary``.
# Callers compare against the ``STATUS_NO_INSTRUCTIONS`` sentinel to decide
# whether to skip writing the (unchanged) text back into the Store.
STATUS_GENERATED = "Generated."
STATUS_REGENERATED = "Regenerated from instructions."
STATUS_APPENDED = "Appended below preserved content."
STATUS_NO_INSTRUCTIONS = (
    "No /prompt instructions — content unchanged. "
    "Add `/prompt …` lines to regenerate or clear the textarea for a fresh draft."
)


# Glossary / convention block appended to every report-agent system prompt.
# The report agent does NOT have tool access to lookup_column, so we have to
# state these conventions in-prompt — the LLM otherwise defaults to common
# business-English readings (e.g. "profit" → company profit) that don't
# match this dataset's column semantics.
_INTERPRETATION_CONVENTIONS = (
    " Interpretation conventions for the figure data: all profit, payout, "
    "bet, win, loss, and RTP metrics are from the PLAYER's perspective — "
    "a negative profit means the player lost money (and the operator / "
    "house earned money on those bets). NEVER describe these as company "
    "profit, casino profit, or house earnings — always frame them as the "
    "player's outcome. When the user message includes a "
    "`# Column definitions` block, treat those definitions as "
    "authoritative and prefer their wording over any prior assumption."
)


_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a data analyst writing a section of an analytical report. "
    "You will receive a JSON summary of one figure (traces, axes, values, "
    "error bars). You may also receive (a) a prior draft to refine, "
    "preserving the user's edits, (b) explicit user instructions to "
    "follow on top of the draft, and (c) reference materials the user "
    "has linked for context. Interpret the figure in 2-3 short paragraphs "
    "of plain prose. Quote specific numbers from the data. Use the "
    "reference materials to ground definitions or contextual claims; do "
    "not invent metrics or dates that are not in the summary or "
    "references. No headings, markdown fences, bullet lists, or HTML — "
    "only paragraphs separated by blank lines. Never echo back "
    "``/prompt`` or ``/prompt:`` lines." + _INTERPRETATION_CONVENTIONS
)


_SUMMARY_SYSTEM_PROMPT = (
    "You are a data analyst writing the overall summary section of an "
    "analytical report. You will receive descriptions and data summaries "
    "for every figure in the report, and may receive (a) a prior draft "
    "to refine, preserving the user's edits, (b) explicit user "
    "instructions to follow, and (c) reference materials the user has "
    "linked for context. Synthesize a 2-3 paragraph overview that ties "
    "the figures together: highlight the main findings, contrasts, and "
    "any trend that appears across multiple figures. Stay grounded in "
    "the provided material — do not invent metrics. No headings, "
    "markdown fences, bullets, or HTML — only paragraphs separated by "
    "blank lines. Never echo back ``/prompt`` or ``/prompt:`` lines." + _INTERPRETATION_CONVENTIONS
)


# Append-mode system prompts: used when the textarea contains preserved
# prose followed by ``/prompt`` lines. The model must NOT echo or rewrite
# the preserved prose — it is concatenated by the caller. The model's job
# is to produce only the new paragraphs to follow.
_DESCRIPTION_APPEND_SYSTEM_PROMPT = (
    "You are a data analyst adding paragraphs to an existing report "
    "section about one figure. The user has prior prose they want "
    "preserved verbatim — your output will be appended directly after "
    "it. DO NOT repeat, paraphrase, restate, or rewrite the preserved "
    "prose; the caller will concatenate it with your output. You will "
    "receive: (a) the preserved prose as read-only context, (b) "
    "explicit user instructions for what to add, (c) the figure's JSON "
    "data summary, and (d) optional reference materials. Generate 1–2 "
    "additional paragraphs of plain prose that follow the instructions "
    "and use the figure data. Quote specific numbers from the data "
    "summary. No headings, markdown fences, bullet lists, HTML, or "
    "``/prompt`` lines. Output only the new paragraphs — the caller "
    "will insert a blank line between the preserved content and your "
    "output." + _INTERPRETATION_CONVENTIONS
)


_SUMMARY_APPEND_SYSTEM_PROMPT = (
    "You are a data analyst adding paragraphs to an existing report "
    "summary that synthesizes across figures. The user has prior prose "
    "they want preserved verbatim — your output will be appended "
    "directly after it. DO NOT repeat, paraphrase, restate, or rewrite "
    "the preserved prose. You will receive: (a) the preserved prose as "
    "read-only context, (b) explicit user instructions for what to "
    "add, (c) every figure's description + data summary, and (d) "
    "optional reference materials. Generate 1–2 additional paragraphs "
    "of plain prose that follow the instructions, draw on the figures' "
    "data, and complement (rather than duplicate) what the preserved "
    "prose already says. No headings, markdown fences, bullets, HTML, "
    "or ``/prompt`` lines. Output only the new paragraphs." + _INTERPRETATION_CONVENTIONS
)


def _invoke_llm(system_prompt: str, user_message: str, *, model_key: Optional[str]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    from ai_agent.chat_agent import _build_llm

    llm = _build_llm(model_key=model_key)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    return response.content if isinstance(response.content, str) else str(response.content)


def _format_prompt_block(prompts: List[str]) -> str:
    return "\n".join(f"- {p}" for p in prompts)


def _format_column_definitions_block(data_blob: str) -> Optional[str]:
    """Build a `# Column definitions` block for any column from
    ``column_metadata.yaml`` whose name appears in the figure data summary.

    The report agent has no tool access to ``lookup_column``, so we
    pre-resolve the relevant subset of the curated metadata and feed it
    inline. Word-boundary regex match keeps the glossary tight to the
    columns the figure actually references.
    """
    try:
        # Reuse the chat agent's process-wide metadata cache so the report
        # endpoints see the same column descriptions as the chat tools.
        from ai_agent.chat_agent import _ensure_metadata

        meta = _ensure_metadata()
    except Exception:
        logger.exception("Could not load column metadata for report glossary")
        return None
    columns = meta.get("columns") or {}
    if not columns:
        return None

    found: List[str] = []
    for name in columns:
        if re.search(rf"\b{re.escape(name)}\b", data_blob):
            found.append(name)
    if not found:
        return None

    lines: List[str] = []
    for name in found:
        info = columns[name]
        desc = (info.get("description") or "").strip()
        lines.append(f"- **{name}**: {desc}")
        formula = info.get("formula")
        if formula:
            lines.append(f"  Formula: `{formula}`")
    return "\n".join(lines)


def _format_references_block(refs: List[Dict[str, str]]) -> Optional[str]:
    """Concatenate fetched references into the ``# Reference materials`` block.

    Inlined here (rather than imported from ``dashboards.report_agent.references``)
    so this module is self-contained — the ai_agent image only needs to
    COPY this file, not the broader ``references.py`` / ``confluence_client.py``
    chain that handles fetching.
    """
    if not refs:
        return None
    parts: List[str] = []
    for ref in refs:
        title = ref.get("title") or ref.get("url") or "(unnamed)"
        url = ref.get("url") or ""
        body = ref.get("text") or ""
        if not body:
            err = ref.get("error") or "(no body fetched — external link)"
            parts.append(f"--- {title} ({url}) ---\n[{err}]")
        else:
            parts.append(f"--- {title} ({url}) ---\n{body}")
    return "\n\n".join(parts)


def generate_description(
    *,
    data_summary: Dict[str, Any],
    language: str = "en",
    model_key: Optional[str] = None,
    existing_description: Optional[str] = None,
    references: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str]:
    """Generate a 2-3 paragraph interpretation of one figure.

    See the module docstring for the four-case behavior. Returns
    ``(text, status)`` where ``status`` is one of the module-level
    ``STATUS_*`` constants; the caller surfaces it as a UI hint and
    can compare against ``STATUS_NO_INSTRUCTIONS`` to skip writing
    the (unchanged) text back into its Store.
    """
    language_name = _language_name(language)
    summary_json = json.dumps(data_summary, ensure_ascii=False, default=str)
    preserved, prompts = _split_at_first_prompt(existing_description)
    is_empty = not (existing_description or "").strip()

    # Case 2: non-empty textarea with no /prompt — no-op so manual edits aren't clobbered.
    if not is_empty and not prompts:
        return (existing_description or ""), STATUS_NO_INSTRUCTIONS

    refs_block = _format_references_block(references or [])
    defs_block = _format_column_definitions_block(summary_json)

    parts: List[str] = [
        f"# Output language\nWrite the response in {language_name}.",
        f"# Figure data\n{summary_json}",
    ]
    if defs_block:
        parts.append(f"# Column definitions\n{defs_block}")
    if refs_block:
        parts.append(f"# Reference materials\n{refs_block}")
    if prompts:
        parts.append(f"# User instructions to follow\n{_format_prompt_block(prompts)}")

    # Case 4: append mode — preserved prose stays verbatim, LLM writes only the new paragraphs.
    if preserved:
        parts.append(f"# Preserved prose (read-only context — do NOT echo or rewrite)\n{preserved}")
        new_content = _invoke_llm(_DESCRIPTION_APPEND_SYSTEM_PROMPT, "\n\n".join(parts), model_key=model_key)
        return preserved + "\n\n" + new_content.strip(), STATUS_APPENDED

    # Cases 1 + 3: full generate. Either empty textarea (first draft) or /prompt at start (instruction-driven regen).
    text = _invoke_llm(_DESCRIPTION_SYSTEM_PROMPT, "\n\n".join(parts), model_key=model_key)
    return text, (STATUS_GENERATED if is_empty else STATUS_REGENERATED)


def generate_summary(
    *,
    figures: List[Dict[str, Any]],
    language: str = "en",
    model_key: Optional[str] = None,
    existing_summary: Optional[str] = None,
    references: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str]:
    """Generate an overall report summary across all figures.

    Same four-case behavior as ``generate_description`` — see module
    docstring. Returns ``(text, status)``.
    """
    language_name = _language_name(language)
    blocks: List[str] = []
    for idx, fig in enumerate(figures, start=1):
        desc = (fig.get("description") or "").strip() or "(no description yet)"
        data = json.dumps(fig.get("data_summary") or {}, ensure_ascii=False, default=str)
        blocks.append(f"## Figure {idx}\nDescription: {desc}\nData: {data}")
    if not blocks:
        return "", STATUS_GENERATED

    preserved, prompts = _split_at_first_prompt(existing_summary)
    is_empty = not (existing_summary or "").strip()

    if not is_empty and not prompts:
        return (existing_summary or ""), STATUS_NO_INSTRUCTIONS

    refs_block = _format_references_block(references or [])
    # Pool every figure's data into one blob so the glossary covers any
    # column appearing in any of them.
    defs_block = _format_column_definitions_block("\n".join(blocks))

    parts: List[str] = [
        f"# Output language\nWrite the response in {language_name}.",
        "# Figures\n" + "\n\n".join(blocks),
    ]
    if defs_block:
        parts.append(f"# Column definitions\n{defs_block}")
    if refs_block:
        parts.append(f"# Reference materials\n{refs_block}")
    if prompts:
        parts.append(f"# User instructions to follow\n{_format_prompt_block(prompts)}")

    if preserved:
        parts.append(f"# Preserved prose (read-only context — do NOT echo or rewrite)\n{preserved}")
        new_content = _invoke_llm(_SUMMARY_APPEND_SYSTEM_PROMPT, "\n\n".join(parts), model_key=model_key)
        return preserved + "\n\n" + new_content.strip(), STATUS_APPENDED

    text = _invoke_llm(_SUMMARY_SYSTEM_PROMPT, "\n\n".join(parts), model_key=model_key)
    return text, (STATUS_GENERATED if is_empty else STATUS_REGENERATED)
