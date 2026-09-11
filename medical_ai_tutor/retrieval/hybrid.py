"""Hybrid retriever: BM25 always, embeddings when available.

Fusion is deterministic — the same corpus and query always produce the same
ranking — because retrieval feeds a pedagogical decision that must be
reproducible in traces and evaluations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..state.models import RetrievedPassage
from .bm25 import BM25Index, tokenize
from .chunking import Chunk
from .embeddings import EmbeddingIndex


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalise into [0, 1]; a flat set maps to 1.0."""
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class HybridRetriever:
    """Searchable-corpus retriever.

    It only ever indexes chunks handed to it by ingestion, which by construction
    excludes protected solutions. It has no code path that can reach the
    protected store.
    """

    def __init__(
        self,
        bm25_weight: float = 0.6,
        embedding_weight: float = 0.4,
        dedup_threshold: float = 0.85,
        use_embeddings: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_score: float = 0.0,
    ) -> None:
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.dedup_threshold = dedup_threshold
        self.use_embeddings = use_embeddings
        self.min_score = min_score

        self.chunks: dict[str, Chunk] = {}
        self.bm25 = BM25Index()
        self.embeddings: EmbeddingIndex | None = None
        if use_embeddings:
            from .embeddings import EmbeddingBackend

            self.embeddings = EmbeddingIndex(EmbeddingBackend(embedding_model))

    # -- construction ---------------------------------------------------- #
    @property
    def embedding_active(self) -> bool:
        return bool(self.embeddings and self.embeddings.available)

    @property
    def mode(self) -> str:
        return "hybrid" if self.embedding_active else "lexical"

    def index(self, chunks: list[Chunk]) -> HybridRetriever:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        payload = [(chunk.chunk_id, self._indexed_text(chunk)) for chunk in chunks]
        self.bm25.build(payload)
        if self.embeddings is not None:
            self.embeddings.build(payload)
        return self

    @staticmethod
    def _indexed_text(chunk: Chunk) -> str:
        # The section title is part of the indexed text: learners often phrase a
        # query in the words of a heading.
        header = " ".join(p for p in [chunk.section_id, chunk.section_title] if p)
        return f"{header}\n{chunk.text}" if header else chunk.text

    # -- search ----------------------------------------------------------- #
    def search(self, query: str, top_k: int = 4) -> list[RetrievedPassage]:
        if not query or not query.strip() or not self.chunks:
            return []

        pool = max(top_k * 4, 12)
        bm25_hits = dict(self.bm25.search(query, top_k=pool))
        embedding_hits: dict[str, float] = {}
        if self.embedding_active:
            embedding_hits = dict(self.embeddings.search(query, top_k=pool))  # type: ignore[union-attr]

        bm25_norm = _normalize(bm25_hits)
        embedding_norm = _normalize(embedding_hits)

        if embedding_norm:
            total = self.bm25_weight + self.embedding_weight
            w_lex = self.bm25_weight / total
            w_dense = self.embedding_weight / total
        else:
            w_lex, w_dense = 1.0, 0.0

        fused: dict[str, float] = {}
        for chunk_id in set(bm25_norm) | set(embedding_norm):
            fused[chunk_id] = (
                w_lex * bm25_norm.get(chunk_id, 0.0) + w_dense * embedding_norm.get(chunk_id, 0.0)
            )

        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

        results: list[RetrievedPassage] = []
        seen_tokens: list[set[str]] = []
        for chunk_id, score in ranked:
            if score < self.min_score:
                continue
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            tokens = set(tokenize(chunk.text))
            if any(_jaccard(tokens, prior) >= self.dedup_threshold for prior in seen_tokens):
                continue  # near-duplicate of a higher-ranked passage
            seen_tokens.append(tokens)
            results.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section_id=chunk.section_id,
                    section_title=chunk.section_title,
                    page=chunk.page,
                    text=chunk.text,
                    score=round(score, 6),
                    bm25_score=round(bm25_hits.get(chunk_id, 0.0), 6),
                    embedding_score=round(embedding_hits.get(chunk_id, 0.0), 6),
                )
            )
            if len(results) >= top_k:
                break
        return results

    # -- persistence ------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for chunk in self.chunks.values():
            kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1
        return {"chunks": len(self.chunks), "mode": self.mode, "kinds": kinds}

    @classmethod
    def from_corpus_file(cls, path: str | Path, **kwargs: Any) -> HybridRetriever:
        """Load a searchable corpus JSONL file produced by ingestion."""
        chunks: list[Chunk] = []
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        chunks.append(Chunk.from_dict(json.loads(line)))
        return cls(**kwargs).index(chunks)
