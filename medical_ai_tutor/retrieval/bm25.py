"""Okapi BM25.

Implemented directly rather than pulled in as a dependency: it is thirty lines of
arithmetic, it must work offline with no model download, and having it in-repo
means the ranking is inspectable and testable.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, domain-aware stop list. Terms like "system" and "information" are load
# bearing in this textbook and are deliberately NOT removed.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here of in on at to for from by
    with without within into onto over under again further is are was were be been being am do
    does did doing have has had having i you he she it we they them his her its our your their
    as such can could should would may might must will shall not no nor only own same so too very
    s t just now also which who whom what when where why how all any both each few more most other
    some
    """.split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    if remove_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


class BM25Index:
    """A BM25 index over a fixed list of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.avg_doc_len: float = 0.0
        self.inverted: dict[str, list[int]] = {}
        self.idf: dict[str, float] = {}

    # -- construction ---------------------------------------------------- #
    def build(self, documents: list[tuple[str, str]]) -> BM25Index:
        """Build from (doc_id, text) pairs."""
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.doc_tokens = [tokenize(text) for _, text in documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

        self.inverted = {}
        for index, freqs in enumerate(self.doc_freqs):
            for term in freqs:
                self.inverted.setdefault(term, []).append(index)

        total_docs = len(self.doc_ids)
        self.idf = {}
        for term, postings in self.inverted.items():
            df = len(postings)
            # BM25+ style floor keeps very common terms from going negative.
            self.idf[term] = max(
                1e-6, math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            )
        return self

    # -- query ------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        if not query_terms or not self.doc_ids:
            return []

        scores: dict[int, float] = {}
        for term in query_terms:
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for index in postings:
                freq = self.doc_freqs[index][term]
                length = self.doc_len[index] or 1
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * (length / (self.avg_doc_len or 1.0))
                )
                scores[index] = scores.get(index, 0.0) + idf * (freq * (self.k1 + 1.0)) / denom

        # Deterministic ordering: score descending, then doc id ascending.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[index], score) for index, score in ranked[:top_k]]

    # -- persistence ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_tokens": self.doc_tokens,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        index.build([(doc_id, " ".join(tokens)) for doc_id, tokens in zip(data["doc_ids"], data["doc_tokens"], strict=True)])
        return index

    def __len__(self) -> int:
        return len(self.doc_ids)
