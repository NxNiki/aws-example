"""
Build new storage HTML from the agent's responses.

All edits go through ``apply_responses`` which takes the original storage and
a list of (anchor, response_html) tuples and returns the new storage with
response fences spliced in. We sort edits in reverse position order so each
splice doesn't invalidate the offsets of later splices.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

from dashboards.report_agent import parser as _parser
from dashboards.report_agent.parser import FigureBlock, PromptBlock


@dataclass
class _Edit:
    """One splice operation, applied right-to-left so positions stay valid."""

    insert_at: int  # position to insert the new fence
    new_text: str
    replace_until: int = -1  # if >= 0, replace storage[insert_at:replace_until] with new_text


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _wrap_response_html(response_text: str) -> str:
    """Convert a plaintext / lightly-marked-up response to Confluence storage HTML.

    Splits on blank lines and wraps each non-empty chunk in ``<p>``. Single
    newlines within a paragraph become ``<br/>``. This is intentionally
    minimal — the LLM is prompted to produce plain prose, not Markdown.
    """
    if not response_text or not response_text.strip():
        return "<p><em>(no response)</em></p>"
    paragraphs = re.split(r"\n\s*\n", response_text.strip())
    out: List[str] = []
    for para in paragraphs:
        clean = _escape_html(para.strip()).replace("\n", "<br/>")
        if clean:
            out.append(f"<p>{clean}</p>")
    return "".join(out) or "<p><em>(no response)</em></p>"


def build_response_fence(response_id: str, input_hash: str, response_text: str) -> str:
    body = _wrap_response_html(response_text)
    return (
        f"<!-- AI-RESPONSE:START id={response_id} hash={input_hash} -->"
        f"{body}"
        f"<!-- AI-RESPONSE:END id={response_id} -->"
    )


def edit_for_figure(figure: FigureBlock, response_text: str) -> _Edit:
    """Insert a response fence right after the FIGURE END marker."""
    input_payload = figure.caption + "|" + str(sorted(figure.filter_context.items()))
    fence = build_response_fence(
        response_id=figure.figure_id,
        input_hash=_hash(input_payload),
        response_text=response_text,
    )
    return _Edit(insert_at=figure.end_pos, new_text=fence)


def edit_for_prompt(prompt: PromptBlock, response_text: str) -> _Edit:
    """
    Replace the prompt's <p> with a 'done'-marked version, then insert the
    response fence directly after it. Encoded as a single splice that
    rewrites both spans together.
    """
    response_id = f"prompt-{_hash(prompt.text + str(prompt.start_pos))}"
    fence = build_response_fence(
        response_id=response_id,
        input_hash=_hash(prompt.text),
        response_text=response_text,
    )
    new_prompt_html = f"<p>/prompt: {_escape_html(prompt.text)} done</p>"
    combined = new_prompt_html + fence
    return _Edit(insert_at=prompt.start_pos, new_text=combined, replace_until=prompt.end_pos)


def apply_edits(storage: str, edits: List[_Edit]) -> str:
    """Apply edits in reverse position order so earlier offsets stay valid."""
    sorted_edits = sorted(edits, key=lambda e: e.insert_at, reverse=True)
    out = storage
    for edit in sorted_edits:
        if edit.replace_until >= 0:
            out = out[: edit.insert_at] + edit.new_text + out[edit.replace_until :]
        else:
            out = out[: edit.insert_at] + edit.new_text + out[edit.insert_at :]
    return out


_FIG_LABEL_REWRITE_RE = re.compile(
    r"(<p[^>]*>\s*<strong>\s*Figure\s+)(\d+)(\s*</strong>\s*</p>)",
    re.IGNORECASE,
)


def _set_figure_label(block_html: str, new_index: int) -> str:
    """Rewrite the visible ``Figure N`` label inside a single FIGURE block."""
    return _FIG_LABEL_REWRITE_RE.sub(rf"\g<1>{new_index}\g<3>", block_html, count=1)


def _figure_unit_span(figure: FigureBlock, parsed: _parser.ParsedDoc, max_gap: int = 200) -> int:
    """Return the end position covering a figure plus its trailing AI-RESPONSE (if any)."""
    for resp in parsed.responses:
        if resp.response_id == figure.figure_id and 0 <= resp.start_pos - figure.end_pos <= max_gap:
            return resp.end_pos
    return figure.end_pos


def reorder_and_renumber_figures(storage: str) -> Optional[str]:
    """Sync FIGURE blocks so visible labels are 1..N consecutively, in document order.

    Two modes, dispatched on the parsed labels:

    - **Permutation mode**: if the labels are a valid permutation of 1..N
      but not already in document order, the user has reordered by editing
      labels. Swap each figure's content (figure + its trailing response) so
      ``Figure 1`` actually appears first, etc., then renumber to 1..N.

    - **Renumber mode**: in any other case (cut-paste reorder, missing/duplicate
      labels, brand-new figures with stale numbers), keep current document
      order and just rewrite the visible labels to 1..N consecutively.

    Returns a new storage string if anything changed, or ``None`` if the doc
    was already canonical so the caller can skip a no-op write.
    """
    parsed = _parser.parse(storage)
    figs = parsed.figures
    if not figs:
        return None

    n = len(figs)
    labels: List[Optional[int]] = [f.index for f in figs]
    expected = list(range(1, n + 1))
    non_none_labels: List[int] = [lbl for lbl in labels if lbl is not None]
    well_formed_permutation = len(non_none_labels) == n and sorted(non_none_labels) == expected

    if well_formed_permutation and non_none_labels != expected:
        # Permutation reorder: pick blocks in label-sorted order
        units_by_label_order = sorted(figs, key=lambda f: f.index or 0)
    else:
        # Cut-paste / first-write / malformed labels: keep doc order
        units_by_label_order = list(figs)

    # Collect each figure's full unit (figure + optional trailing response) span/html.
    units = [
        (f.start_pos, _figure_unit_span(f, parsed), storage[f.start_pos : _figure_unit_span(f, parsed)]) for f in figs
    ]

    desired_units_in_doc_order: List[str] = []
    for new_idx, src_fig in enumerate(units_by_label_order, start=1):
        src_start = src_fig.start_pos
        src_end = _figure_unit_span(src_fig, parsed)
        src_html = storage[src_start:src_end]
        desired_units_in_doc_order.append(_set_figure_label(src_html, new_idx))

    if [u[2] for u in units] == desired_units_in_doc_order:
        return None  # already canonical

    edits: List[_Edit] = []
    for (orig_start, orig_end, _orig_html), new_html in zip(units, desired_units_in_doc_order):
        edits.append(_Edit(insert_at=orig_start, new_text=new_html, replace_until=orig_end))
    return apply_edits(storage, edits)
