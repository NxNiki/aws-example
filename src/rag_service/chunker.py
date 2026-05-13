"""
Sliding-window text chunker.

Operates on character counts (not tokens) to avoid pulling tiktoken into the
service image. ~4 chars ≈ 1 token for English, so chunk_chars=1200 gives
~300 tokens per chunk — comfortable for embedding models with 8K-token limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    page_id: str
    title: str
    chunk_index: int
    text: str
    url: str = ""
    space_key: str = ""
    source_name: str = ""


def chunk_text(
    page_id: str,
    title: str,
    text: str,
    *,
    url: str = "",
    space_key: str = "",
    source_name: str = "",
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> List[Chunk]:
    """Split ``text`` into overlapping windows of ~``chunk_chars`` characters."""
    if not text or not text.strip():
        return []
    if overlap >= chunk_chars:
        raise ValueError("overlap must be smaller than chunk_chars")

    chunks: List[Chunk] = []
    step = chunk_chars - overlap
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + chunk_chars].strip()
        if piece:
            chunks.append(
                Chunk(
                    page_id=page_id,
                    title=title,
                    chunk_index=i,
                    text=piece,
                    url=url,
                    space_key=space_key,
                    source_name=source_name,
                )
            )
        if start + chunk_chars >= len(text):
            break
    return chunks
