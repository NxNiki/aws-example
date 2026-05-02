"""
Parse a Confluence page's storage HTML into the structured blocks defined in
``docs/report_agent.md``: figures, prompts, response fences, and any
Confluence links present in the body.

Storage format is XHTML-ish with custom ``ac:``/``ri:`` tags. We use
position-anchored regexes rather than a real XML parser because:
  * Confluence's storage format isn't strictly XML (custom namespaces, mixed
    HTML-comment markers, occasional unbalanced tags from the WYSIWYG editor),
  * we only need to find a few well-formed marker pairs we own,
  * positions are stable when the surrounding prose changes, making splicing
    deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import unescape
from typing import Dict, List, Optional


@dataclass
class FigureBlock:
    figure_id: str
    raw_html: str
    start_pos: int
    end_pos: int  # exclusive (just past the END marker)
    caption: str
    filter_context: Dict[str, object]
    index: Optional[int] = None  # parsed from the visible "Figure N" label


@dataclass
class PromptBlock:
    text: str  # the prompt body without the trailing " done" marker
    raw_html: str  # the full enclosing <p>...</p>
    start_pos: int
    end_pos: int
    is_done: bool


@dataclass
class ResponseBlock:
    response_id: str
    start_pos: int
    end_pos: int


@dataclass
class ParsedDoc:
    storage: str
    figures: List[FigureBlock] = field(default_factory=list)
    prompts: List[PromptBlock] = field(default_factory=list)
    responses: List[ResponseBlock] = field(default_factory=list)
    confluence_links: List[str] = field(default_factory=list)


_FIGURE_RE = re.compile(
    r"<!-- FIGURE:START id=([0-9a-fA-F]+) -->(.*?)<!-- FIGURE:END id=\1 -->",
    re.DOTALL,
)
_RESPONSE_RE = re.compile(
    r"<!-- AI-RESPONSE:START id=([^\s>]+)[^>]*-->(.*?)<!-- AI-RESPONSE:END id=\1 -->",
    re.DOTALL,
)
_PROMPT_RE = re.compile(
    r"<p[^>]*>\s*/prompt:\s*(.*?)\s*</p>",
    re.DOTALL | re.IGNORECASE,
)
_HREF_RE = re.compile(r'<a[^>]+href="([^"]+)"', re.IGNORECASE)
_CAPTION_RE = re.compile(r"<strong>\s*Caption:\s*</strong>\s*(.*?)</p>", re.DOTALL | re.IGNORECASE)
_CONTEXT_RE = re.compile(r"<strong>\s*Context:\s*</strong>\s*<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
_FIG_LABEL_RE = re.compile(
    r"<p[^>]*>\s*<strong>\s*Figure\s+(\d+)\s*</strong>\s*</p>",
    re.IGNORECASE,
)


def _parse_figure_index(raw_html: str) -> Optional[int]:
    match = _FIG_LABEL_RE.search(raw_html)
    return int(match.group(1)) if match else None


def _parse_filter_context(raw_html: str) -> Dict[str, object]:
    match = _CONTEXT_RE.search(raw_html)
    if not match:
        return {}
    try:
        return json.loads(unescape(match.group(1)))
    except json.JSONDecodeError:
        return {}


def _parse_caption(raw_html: str) -> str:
    match = _CAPTION_RE.search(raw_html)
    if not match:
        return ""
    return unescape(match.group(1)).strip()


def parse(storage: str) -> ParsedDoc:
    doc = ParsedDoc(storage=storage)

    for match in _FIGURE_RE.finditer(storage):
        body = match.group(2)
        doc.figures.append(
            FigureBlock(
                figure_id=match.group(1),
                raw_html=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
                caption=_parse_caption(body),
                filter_context=_parse_filter_context(body),
                index=_parse_figure_index(body),
            )
        )

    for match in _RESPONSE_RE.finditer(storage):
        doc.responses.append(
            ResponseBlock(
                response_id=match.group(1),
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    for match in _PROMPT_RE.finditer(storage):
        body = match.group(1)
        is_done = bool(re.search(r"\bdone\s*$", body, re.IGNORECASE))
        text = re.sub(r"\bdone\s*$", "", body, flags=re.IGNORECASE).strip() if is_done else body.strip()
        doc.prompts.append(
            PromptBlock(
                text=text,
                raw_html=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
                is_done=is_done,
            )
        )

    doc.confluence_links = [match.group(1) for match in _HREF_RE.finditer(storage)]
    return doc


def find_response_after(doc: ParsedDoc, prompt: PromptBlock, *, max_gap: int = 200) -> Optional[ResponseBlock]:
    """Return the response block sitting immediately after ``prompt`` (within ``max_gap`` chars)."""
    for resp in doc.responses:
        if 0 <= resp.start_pos - prompt.end_pos <= max_gap:
            return resp
    return None


def figure_response(doc: ParsedDoc, figure: FigureBlock) -> Optional[ResponseBlock]:
    """Return the response block matching ``figure.figure_id``, if any."""
    for resp in doc.responses:
        if resp.response_id == figure.figure_id:
            return resp
    return None


def pending_figures(doc: ParsedDoc) -> List[FigureBlock]:
    return [f for f in doc.figures if figure_response(doc, f) is None]


def pending_prompts(doc: ParsedDoc) -> List[PromptBlock]:
    return [p for p in doc.prompts if not p.is_done]
