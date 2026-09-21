"""Markdown -> retrievable chunks.

Splitting on headings first (instead of blindly every N characters) keeps a
concept and its explanation in the same chunk, which is most of what makes
retrieval feel accurate.
"""
from __future__ import annotations

import re
from pathlib import Path

from config.settings import settings

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(heading, body)] preserving document order."""
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("", text.strip())]

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            sections.append((m.group(2).strip(), body))
    return sections


def _window(body: str, size: int, overlap: int) -> list[str]:
    """Character windows that try to break on sentence boundaries."""
    if len(body) <= size:
        return [body]
    out, start = [], 0
    while start < len(body):
        end = min(start + size, len(body))
        if end < len(body):
            pivot = body.rfind(". ", start + size // 2, end)
            if pivot != -1:
                end = pivot + 1
        out.append(body[start:end].strip())
        if end >= len(body):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def chunk_file(path: Path) -> list[dict[str, str]]:
    """Chunk one markdown file into records ready for embedding."""
    text = path.read_text(encoding="utf-8")
    chunks: list[dict[str, str]] = []

    for heading, body in split_sections(text):
        for i, piece in enumerate(_window(body, settings.chunk_size, settings.chunk_overlap)):
            # Prefixing the heading gives the embedder topical context it
            # would otherwise lose when a section is split.
            content = f"{heading}\n{piece}" if heading else piece
            chunks.append({
                "id": f"{path.stem}::{heading or 'intro'}::{i}",
                "text": content.strip(),
                "title": heading or path.stem.replace("_", " ").title(),
                "source": path.name,
            })
    return chunks


def chunk_directory(directory: Path | None = None) -> list[dict[str, str]]:
    directory = directory or settings.kb_dir
    records: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.md")):
        records.extend(chunk_file(path))
    return records
