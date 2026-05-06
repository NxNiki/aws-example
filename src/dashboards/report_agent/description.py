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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "zh-Hans": "Simplified Chinese",
    "zh-Hant": "Traditional Chinese",
}


def _language_name(language_code: str) -> str:
    return SUPPORTED_LANGUAGES.get(language_code, "English")


_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a data analyst writing a section of an analytical report. "
    "You will receive a JSON summary of one figure (traces, axes, values, "
    "error bars). Interpret it in 2-3 short paragraphs of plain prose. "
    "Quote specific numbers from the data. Do not invent metrics or dates "
    "that are not in the summary. Do not include headings, markdown fences, "
    "bullet lists, or HTML — only paragraphs separated by blank lines."
)


_SUMMARY_SYSTEM_PROMPT = (
    "You are a data analyst writing the overall summary section of an "
    "analytical report. You will receive descriptions and data summaries for "
    "every figure in the report. Synthesize a 2-3 paragraph overview that "
    "ties them together: highlight the main findings, contrasts, and any "
    "trend that appears across multiple figures. Stay grounded in the "
    "provided material — do not invent metrics. No headings, markdown "
    "fences, bullets, or HTML — only paragraphs separated by blank lines."
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
    """Generate a 2-3 paragraph interpretation of one figure in the requested language."""
    language_name = _language_name(language)
    summary_json = json.dumps(data_summary, ensure_ascii=False, default=str)
    parts: List[str] = [
        f"# Output language\nWrite the response in {language_name}.",
        f"# Figure data\n{summary_json}",
    ]
    if existing_description:
        parts.append(f"# Previous description (refine or replace)\n{existing_description}")
    return _invoke_llm(_DESCRIPTION_SYSTEM_PROMPT, "\n\n".join(parts), model_key=model_key)


def generate_summary(
    *,
    figures: List[Dict[str, Any]],
    language: str = "en",
    model_key: Optional[str] = None,
) -> str:
    """Generate an overall report summary across all figures.

    ``figures`` is the list stored in the Report tab's ``report-figures``
    store: each item must have ``data_summary`` and may have ``description``.
    """
    language_name = _language_name(language)
    blocks: List[str] = []
    for idx, fig in enumerate(figures, start=1):
        desc = (fig.get("description") or "").strip() or "(no description yet)"
        data = json.dumps(fig.get("data_summary") or {}, ensure_ascii=False, default=str)
        blocks.append(f"## Figure {idx}\nDescription: {desc}\nData: {data}")
    if not blocks:
        return ""
    user_message = f"# Output language\nWrite the response in {language_name}.\n\n# Figures\n" + "\n\n".join(blocks)
    return _invoke_llm(_SUMMARY_SYSTEM_PROMPT, user_message, model_key=model_key)
