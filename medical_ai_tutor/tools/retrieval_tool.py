"""search_course_material - retrieval over the searchable corpus only."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..retrieval.hybrid import HybridRetriever
from .base import Tool


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)


class SearchCourseMaterialTool(Tool):
    name = "search_course_material"
    description = (
        "Search the course textbook, glossary and exercise questions for passages "
        "relevant to a query. Returns passages with section and page citations. "
        "Contains no official solutions."
    )
    args_schema = SearchArgs

    def __init__(self, retriever: HybridRetriever, max_passage_chars: int = 1100) -> None:
        self.retriever = retriever
        self.max_passage_chars = max_passage_chars

    def run(self, args: SearchArgs) -> dict[str, Any]:  # type: ignore[override]
        passages = self.retriever.search(args.query, top_k=args.top_k)
        return {
            "query": args.query,
            "mode": self.retriever.mode,
            "passages": [
                {
                    "chunk_id": p.chunk_id,
                    "section_id": p.section_id,
                    "section_title": p.section_title,
                    "page": p.page,
                    "citation": p.citation(),
                    "text": p.text[: self.max_passage_chars],
                    "score": p.score,
                }
                for p in passages
            ],
        }
