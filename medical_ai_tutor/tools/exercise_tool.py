"""get_exercise - public exercise metadata.

This tool reads `data/exercises/`. It has no path, import, or configuration that
reaches `data/protected/`, and the payload it returns has no field capable of
carrying solution text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..state.models import ExerciseMeta
from .base import Tool, ToolError

# Fields that must never appear in this tool's output, whatever the data file
# happens to contain.
_FORBIDDEN_FIELDS = frozenset({"solution", "solution_text", "answer", "answer_units", "official"})


class ExerciseCatalog:
    """In-memory catalogue of public exercise metadata."""

    def __init__(self, exercises: list[ExerciseMeta] | None = None) -> None:
        self._by_id: dict[str, ExerciseMeta] = {e.exercise_id: e for e in exercises or []}

    @classmethod
    def from_file(cls, path: str | Path) -> ExerciseCatalog:
        path = Path(path)
        if not path.exists():
            return cls([])
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)

        exercises: list[ExerciseMeta] = []
        for entry in raw:
            leaked = _FORBIDDEN_FIELDS & set(entry)
            if leaked:
                # Defensive: the exercises file is public data. If a future
                # ingestion bug ever writes solution text here, fail loudly
                # rather than serve it.
                raise ToolError(
                    f"exercise {entry.get('exercise_id')} carries protected field(s): "
                    f"{sorted(leaked)}"
                )
            exercises.append(
                ExerciseMeta(
                    exercise_id=entry["exercise_id"],
                    title=entry.get("title", ""),
                    question=entry.get("question", ""),
                    chapter=entry.get("chapter"),
                    concepts=entry.get("concepts", []),
                    protected_unit_count=int(entry.get("protected_unit_count", 0)),
                )
            )
        return cls(exercises)

    def get(self, exercise_id: str) -> ExerciseMeta | None:
        return self._by_id.get(exercise_id)

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def all(self) -> list[ExerciseMeta]:
        return [self._by_id[key] for key in self.ids()]

    def is_protected(self, exercise_id: str | None) -> bool:
        """An exercise is protected when an official solution exists for it."""
        if not exercise_id:
            return False
        meta = self._by_id.get(exercise_id)
        return bool(meta and meta.protected_unit_count > 0)

    def __len__(self) -> int:
        return len(self._by_id)


class GetExerciseArgs(BaseModel):
    exercise_id: str = Field(min_length=1, max_length=32)


class GetExerciseTool(Tool):
    name = "get_exercise"
    description = (
        "Fetch the question text and public metadata of a course exercise by id "
        "(for example '2.16.1'). Never returns the official solution."
    )
    args_schema = GetExerciseArgs

    def __init__(self, catalog: ExerciseCatalog) -> None:
        self.catalog = catalog

    def run(self, args: GetExerciseArgs) -> dict[str, Any]:  # type: ignore[override]
        meta = self.catalog.get(args.exercise_id)
        if meta is None:
            raise ToolError(
                f"unknown exercise {args.exercise_id!r}; known ids: {self.catalog.ids()[:8]}"
            )
        return {
            "exercise_id": meta.exercise_id,
            "title": meta.title,
            "question": meta.question,
            "chapter": meta.chapter,
            "concepts": meta.concepts,
            "protected": meta.protected_unit_count > 0,
        }
