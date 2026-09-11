"""Optional dense retrieval.

Embeddings are a ranking improvement, never a requirement. If
`sentence-transformers` is not installed or the model cannot be loaded (no
network, no cache), the backend reports itself unavailable and the hybrid
retriever runs lexical-only. Tests therefore never depend on a model download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingBackend:
    """Thin wrapper around a sentence-transformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None
        self.unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        if self._model is not None:
            return True
        if self.unavailable_reason is not None:
            return False
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception as exc:  # ImportError, network failure, missing cache
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            return False

    def encode(self, texts: list[str]) -> np.ndarray | None:
        if not texts or not self.available:
            return None
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class EmbeddingIndex:
    """Cosine-similarity index over pre-normalised vectors."""

    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        self.backend = backend or EmbeddingBackend()
        self.doc_ids: list[str] = []
        self.matrix: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return self.matrix is not None and len(self.doc_ids) > 0

    def build(self, documents: list[tuple[str, str]]) -> EmbeddingIndex:
        if not documents or not self.backend.available:
            self.doc_ids, self.matrix = [], None
            return self
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.matrix = self.backend.encode([text for _, text in documents])
        return self

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.available:
            return []
        query_vector = self.backend.encode([query])
        if query_vector is None:
            return []
        scores = (self.matrix @ query_vector[0]).tolist()
        ranked = sorted(
            zip(self.doc_ids, scores, strict=True), key=lambda pair: (-pair[1], pair[0])
        )
        return ranked[:top_k]

    def save(self, path: str | Path) -> None:
        if not self.available:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), matrix=self.matrix)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as handle:
            json.dump({"doc_ids": self.doc_ids, "model": self.backend.model_name}, handle)

    def load(self, path: str | Path) -> EmbeddingIndex:
        path = Path(path)
        meta_path, matrix_path = path.with_suffix(".json"), path.with_suffix(".npz")
        if not meta_path.exists() or not matrix_path.exists():
            return self
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        self.doc_ids = meta.get("doc_ids", [])
        self.matrix = np.load(matrix_path)["matrix"]
        return self
