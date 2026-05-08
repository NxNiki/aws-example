"""
LLM helpers for the Report tab.

Two entry points:
  * ``generate_description`` — interpret a single figure given its
    ``data_summary`` (and optionally an existing description the user wants
    refined). Returns prose in the requested language.
  * ``generate_summary`` — synthesize an overall report summary across all
    figures' descriptions + data summaries.

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


# Lines starting with ``/prompt:`` are user instructions, not prose to keep.
# Anywhere in the textarea, one per line, case-insensitive.
_PROMPT_LINE_RE = re.compile(r"^[ \t]*/prompt:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _split_prompts(text: Optional[str]) -> Tuple[str, List[str]]:
    """Split a textarea value into ``(prose_to_preserve, list_of_user_instructions)``.

    Empty string or ``None`` returns ``("", [])``.
    """
    if not text:
        return "", []
    prompts = [m.group(1).strip() for m in _PROMPT_LINE_RE.finditer(text)]
    prose = _PROMPT_LINE_RE.sub("", text).strip()
    return prose, prompts


_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a data analyst writing a section of an analytical report. "
    "You will receive a JSON summary of one figure (traces, axes, values, "
    "error bars). You may also receive (a) a prior draft to refine, "
    "preserving the user's edits, and (b) explicit user instructions to "
    "follow on top of the draft. Interpret the figure in 2-3 short "
    "paragraphs of plain prose. Quote specific numbers from the data. Do "
    "not invent metrics or dates that are not in the summary. No headings, "
    "markdown fences, bullet lists, or HTML — only paragraphs separated "
    "by blank lines. Never echo back ``/prompt:`` lines."
)


_SUMMARY_SYSTEM_PROMPT = (
    "You are a data analyst writing the overall summary section of an "
    "analytical report. You will receive descriptions and data summaries "
    "for every figure in the report, and may receive (a) a prior draft "
    "to refine, preserving the user's edits, and (b) explicit user "
    "instructions to follow. Synthesize a 2-3 paragraph overview that "
    "ties the figures together: highlight the main findings, contrasts, "
    "and any trend that appears across multiple figures. Stay grounded "
    "in the provided material — do not invent metrics. No headings, "
    "markdown fences, bullets, or HTML — only paragraphs separated by "
    "blank lines. Never echo back ``/prompt:`` lines."
)


def _invoke_llm(system_prompt: str, user_message: str, *, model_key: Optional[str]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    from dashboards.chat_agent import _build_llm

    llm = _build_llm(model_key=model_key)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    return response.content if isinstance(response.content, str) else str(response.content)


def generate_description(
    *,
    data_summary: Dict[str, Any],
    language: str = "en",
    model_key: Optional[str] = None,
    existing_description: Optional[str] = None,
) -> str:
    """Generate a 2-3 paragraph interpretation of one figure.

    ``existing_description`` may contain prior LLM prose mixed with user
    edits and ``/prompt: ...`` instruction lines. Prose is sent to the LLM
    as a draft to preserve, prompts as explicit instructions to follow.
    """
    language_name = _language_name(language)
    summary_json = json.dumps(data_summary, ensure_ascii=False, default=str)
    prior_prose, prompts = _split_prompts(existing_description)
    parts: List[str] = [
        f"# Output language\nWrite the response in {language_name}.",
        f"# Figure data\n{summary_json}",
    ]
    if prior_prose:
        parts.append(f"# Prior draft (refine, preserving any user edits)\n{prior_prose}")
    if prompts:
        prompt_block = "\n".join(f"- {p}" for p in prompts)
        parts.append(f"# User instructions to follow\n{prompt_block}")
    return _invoke_llm(_DESCRIPTION_SYSTEM_PROMPT, "\n\n".join(parts), model_key=model_key)


def generate_summary(
    *,
    figures: List[Dict[str, Any]],
    language: str = "en",
    model_key: Optional[str] = None,
    existing_summary: Optional[str] = None,
) -> str:
    """Generate an overall report summary across all figures.

    ``figures`` is the list stored in the Report tab's ``report-figures``
    store: each item must have ``data_summary`` and may have ``description``.
    ``existing_summary`` works the same way as ``existing_description`` —
    prose to preserve, plus optional ``/prompt:`` instruction lines.
    """
    language_name = _language_name(language)
    blocks: List[str] = []
    for idx, fig in enumerate(figures, start=1):
        desc = (fig.get("description") or "").strip() or "(no description yet)"
        data = json.dumps(fig.get("data_summary") or {}, ensure_ascii=False, default=str)
        blocks.append(f"## Figure {idx}\nDescription: {desc}\nData: {data}")
    if not blocks:
        return ""
    parts: List[str] = [
        f"# Output language\nWrite the response in {language_name}.",
        "# Figures\n" + "\n\n".join(blocks),
    ]
    prior_prose, prompts = _split_prompts(existing_summary)
    if prior_prose:
        parts.append(f"# Prior draft (refine, preserving any user edits)\n{prior_prose}")
    if prompts:
        prompt_block = "\n".join(f"- {p}" for p in prompts)
        parts.append(f"# User instructions to follow\n{prompt_block}")
    return _invoke_llm(_SUMMARY_SYSTEM_PROMPT, "\n\n".join(parts), model_key=model_key)
