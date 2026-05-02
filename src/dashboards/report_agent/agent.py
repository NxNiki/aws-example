"""
Update-doc orchestrator.

End-to-end flow on each ``Update doc`` click:
  1. Read the page (storage + version).
  2. Parse out figures, prompts, response fences, and Confluence links.
  3. If nothing is pending, return a no-op result.
  4. Load reference pages depth-2 from every Confluence link in the doc.
  5. For each pending request, build a one-shot LLM prompt with shared
     context + the request's specifics + responses already written above
     this point in document order.
  6. Splice all responses into a new storage body.
  7. Save the page back.

Each pending request gets one LLM call. We deliberately keep this loop
simple — no langgraph, no tools — for v1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

from dashboards import confluence_client
from dashboards.report_agent import parser, references, writer

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    status: str  # 'ok' | 'noop' | 'error'
    processed: int = 0
    figures_processed: int = 0
    prompts_processed: int = 0
    references_loaded: int = 0
    message: str = ""


_SYSTEM_PROMPT = (
    "You are a data analyst writing a section of an analytical report inside a "
    "Confluence document. The user has dropped figures into the doc and you must "
    "interpret them, or has asked questions inline that you must answer. "
    "Respond in 2-3 short paragraphs of plain prose. Do NOT include headings, "
    "markdown fences, bullet lists, or HTML — only paragraphs separated by blank "
    "lines. Stay grounded in the figure's filter context and the reference "
    "materials. Do not invent metrics, dates, or numbers that are not in the "
    "provided context."
)


def _format_figure_request(figure: parser.FigureBlock) -> str:
    ctx_lines = "\n".join(f"  {k}: {v}" for k, v in figure.filter_context.items())
    cap = figure.caption.strip() or "(no caption — interpret from filter context alone)"
    label = f"Figure {figure.index}" if figure.index is not None else "An unlabelled figure"
    return f"TASK: Interpret {label} for the report.\n" f"Caption: {cap}\n" f"Filter context:\n{ctx_lines}"


def _format_prompt_request(prompt: parser.PromptBlock) -> str:
    return f"TASK: Answer the following inline question:\n{prompt.text}"


def _build_user_message(
    *,
    intro: str,
    refs_block: Optional[str],
    figure_index_block: Optional[str],
    prior_responses: List[str],
    request_section: str,
) -> str:
    parts: List[str] = []
    if intro:
        parts.append(f"# Document intro\n{intro}")
    if figure_index_block:
        parts.append(f"# Figures present in this doc (in order)\n{figure_index_block}")
    if refs_block:
        parts.append(f"# Reference materials\n{refs_block}")
    if prior_responses:
        parts.append("# Earlier sections of this report (already written)\n" + "\n\n".join(prior_responses))
    parts.append(f"# Your task\n{request_section}")
    return "\n\n".join(parts)


def _figure_index_block(parsed: parser.ParsedDoc) -> Optional[str]:
    """A short table of `Figure N: caption` so /prompt: lines can reference figures by number."""
    if not parsed.figures:
        return None
    lines = []
    for fig in parsed.figures:
        label = f"Figure {fig.index}" if fig.index is not None else "Figure ?"
        cap = fig.caption.strip() or "(no caption)"
        lines.append(f"- {label}: {cap}")
    return "\n".join(lines)


def _extract_intro(parsed: parser.ParsedDoc) -> str:
    """Best-effort: text before the first request block, stripped of HTML."""
    earliest = min(
        [b.start_pos for b in parsed.figures]
        + [p.start_pos for p in parsed.prompts]
        + [r.start_pos for r in parsed.responses]
        + [len(parsed.storage)]
    )
    intro_html = parsed.storage[:earliest]
    from dashboards.confluence_client import _strip_html

    return _strip_html(intro_html).strip()


def _ordered_pending(parsed: parser.ParsedDoc) -> List[Union[parser.FigureBlock, parser.PromptBlock]]:
    """Pending requests sorted by document position so prompts above land first."""
    pending: List[Union[parser.FigureBlock, parser.PromptBlock]] = []
    pending.extend(parser.pending_figures(parsed))
    pending.extend(parser.pending_prompts(parsed))
    pending.sort(key=lambda req: req.start_pos)
    return pending


def update_doc(
    page_url: str,
    *,
    model_key: Optional[str] = None,
) -> UpdateResult:
    """The Update Doc entry point. Returns a structured result for the UI."""
    page_id = confluence_client.extract_page_id_from_url(page_url)
    if not page_id:
        return UpdateResult(status="error", message=f"Could not parse a page ID from URL: {page_url!r}")

    page = confluence_client.get_page_storage(page_id)
    storage = page["storage"]

    # Reorder/renumber figures BEFORE finding pending requests, so labels and
    # positions are canonical in this pass and the LLM sees the same numbering
    # the user will see after save.
    reordered = writer.reorder_and_renumber_figures(storage)
    if reordered is not None:
        storage = reordered

    parsed = parser.parse(storage)
    pending = _ordered_pending(parsed)
    figures_renumbered = reordered is not None

    if not pending and not figures_renumbered:
        return UpdateResult(status="noop", message="Nothing pending — no new figures or active /prompt: lines.")

    refs = references.load_references(storage)
    refs_block = references.format_for_prompt(refs)
    intro = _extract_intro(parsed)
    figure_block = _figure_index_block(parsed)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from dashboards.chat_agent import _build_llm
    except ImportError as exc:
        return UpdateResult(status="error", message=f"LLM deps missing: {exc}")

    llm = _build_llm(model_key=model_key)

    edits: List = []
    prior_responses: List[str] = []
    figures_processed = 0
    prompts_processed = 0

    for request in pending:
        if isinstance(request, parser.FigureBlock):
            request_section = _format_figure_request(request)
        else:
            request_section = _format_prompt_request(request)

        user_msg = _build_user_message(
            intro=intro,
            refs_block=refs_block,
            figure_index_block=figure_block,
            prior_responses=prior_responses,
            request_section=request_section,
        )
        try:
            ai_response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)])
        except Exception as exc:
            logger.exception("LLM call failed for request at pos=%s", request.start_pos)
            return UpdateResult(status="error", message=f"LLM call failed: {exc}")

        text = ai_response.content if isinstance(ai_response.content, str) else str(ai_response.content)
        prior_responses.append(text)

        if isinstance(request, parser.FigureBlock):
            edits.append(writer.edit_for_figure(request, text))
            figures_processed += 1
        else:
            edits.append(writer.edit_for_prompt(request, text))
            prompts_processed += 1

    new_storage = writer.apply_edits(storage, edits)
    confluence_client.update_page_storage(
        page_id,
        title=page["title"],
        new_storage=new_storage,
        expected_version=page["version"],
    )
    extras = []
    if figures_renumbered:
        extras.append("renumbered figures")
    if pending:
        extras.append(f"processed {len(pending)} request(s)")
    extras.append(f"loaded {len(refs)} reference page(s)")
    return UpdateResult(
        status="ok",
        processed=len(pending),
        figures_processed=figures_processed,
        prompts_processed=prompts_processed,
        references_loaded=len(refs),
        message="; ".join(extras) + ".",
    )
