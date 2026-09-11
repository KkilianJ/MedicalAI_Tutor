"""Section-aware chunking.

Chunks follow the book's own section structure instead of a blind sliding
window, so a retrieved passage is a coherent piece of one section and carries the
section id, title and page needed to cite it back to the learner.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Sentence boundary that does not fire after common abbreviations, so
# "Mr. Russo" and "e.g. HL7" stay in one piece.
_SENTENCE_RE = re.compile(r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bProf\.)(?<!\bFig\.)(?<!\bNo\.)(?<!\bvs\.)(?<!\bcf\.)(?<!\betc\.)(?<!\bal\.)(?<!\bSect\.)(?<!\bChap\.)(?<!\bapprox\.)(?<!\be\.g\.)(?<!\bi\.e\.)" r"(?<=[.!?])\s+(?=[A-Z(])")


def approx_tokens(text: str) -> int:
    """Whitespace/word count as a stand-in for a tokenizer.

    Deliberately dependency-free: chunk sizing does not need to match any
    particular BPE, only to be stable and roughly right.
    """
    return len(_WORD_RE.findall(text))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section_id: str | None = None
    section_title: str | None = None
    page: int | None = None
    kind: str = "textbook"  # textbook | glossary | exercise
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "page": self.page,
            "kind": self.kind,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            text=data["text"],
            section_id=data.get("section_id"),
            section_title=data.get("section_title"),
            page=data.get("page"),
            kind=data.get("kind", "textbook"),
            metadata=data.get("metadata", {}),
        )


def make_chunk_id(doc_id: str, section_id: str | None, index: int, text: str) -> str:
    """Stable id: re-ingesting identical content yields identical ids."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}:{section_id or 'na'}:{index:03d}:{digest}"


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_section(
    text: str,
    doc_id: str,
    section_id: str | None,
    section_title: str | None,
    page: int | None = None,
    target_tokens: int = 220,
    overlap_tokens: int = 40,
    min_tokens: int = 40,
    kind: str = "textbook",
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split one section into overlapping, sentence-aligned chunks."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    index = 0

    def flush(force: bool = False) -> None:
        nonlocal buffer, buffer_tokens, index
        if not buffer:
            return
        body = " ".join(buffer).strip()
        if not force and approx_tokens(body) < min_tokens and chunks:
            # Too small to stand alone: append to the previous chunk instead.
            previous = chunks[-1]
            previous.text = f"{previous.text} {body}".strip()
            buffer, buffer_tokens = [], 0
            return
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc_id, section_id, index, body),
                doc_id=doc_id,
                text=body,
                section_id=section_id,
                section_title=section_title,
                page=page,
                kind=kind,
                metadata=dict(metadata or {}),
            )
        )
        index += 1
        # Carry a sentence-aligned tail forward so ideas spanning a boundary are
        # still retrievable.
        if overlap_tokens > 0:
            tail: list[str] = []
            tail_tokens = 0
            for sentence in reversed(buffer):
                tokens = approx_tokens(sentence)
                if tail_tokens + tokens > overlap_tokens:
                    break
                tail.insert(0, sentence)
                tail_tokens += tokens
            buffer, buffer_tokens = tail, tail_tokens
        else:
            buffer, buffer_tokens = [], 0

    for sentence in sentences:
        tokens = approx_tokens(sentence)
        if buffer_tokens + tokens > target_tokens and buffer:
            flush()
        buffer.append(sentence)
        buffer_tokens += tokens

    flush(force=not chunks)
    return chunks


def chunk_documents(
    documents: Iterable[dict[str, Any]],
    target_tokens: int = 220,
    overlap_tokens: int = 40,
    min_tokens: int = 40,
) -> list[Chunk]:
    """Chunk a stream of section documents produced by ingestion."""
    out: list[Chunk] = []
    for doc in documents:
        out.extend(
            chunk_section(
                text=doc.get("text", ""),
                doc_id=doc.get("doc_id", "doc"),
                section_id=doc.get("section_id"),
                section_title=doc.get("section_title"),
                page=doc.get("page"),
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                min_tokens=min_tokens,
                kind=doc.get("kind", "textbook"),
                metadata=doc.get("metadata", {}),
            )
        )
    return out
