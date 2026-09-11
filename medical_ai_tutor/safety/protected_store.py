"""Isolated store for official exercise solutions.

This is the only module in the system that reads `data/protected/`. Nothing in
`tutor/`, `tools/`, `retrieval/` or `app/` imports it; only `safety/` does.

Every read is logged. The evaluation suite asserts that the access log contains
no entry attributed to a non-safety component, which turns "protected solutions
are isolated" from a claim into a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Components permitted to read a protected solution. Anything else is an
# architectural violation and is recorded as such.
ALLOWED_ACCESSORS = frozenset({"safety_gate", "semantic_judge", "deterministic_detector", "test"})


class ProtectedAccessViolation(RuntimeError):
    """Raised when a non-safety component attempts to read a solution."""


@dataclass
class ProtectedSolution:
    exercise_id: str
    title: str
    solution_text: str
    answer_units: list[str] = field(default_factory=list)
    page: int | None = None

    def unit_labels(self) -> list[str]:
        """Short, non-revealing labels for the answer units.

        Safe to put in a trace or a verdict; the unit text itself is not.
        """
        labels: list[str] = []
        for index, unit in enumerate(self.answer_units, start=1):
            words = unit.split()
            head = " ".join(words[:4])
            labels.append(f"{self.exercise_id}#u{index}:{head}…")
        return labels


class ProtectedStore:
    """Read-only access to official solutions, restricted to safety callers."""

    def __init__(self, solutions: list[ProtectedSolution] | None = None) -> None:
        self._by_id: dict[str, ProtectedSolution] = {
            s.exercise_id: s for s in solutions or []
        }
        self.access_log: list[dict[str, Any]] = []

    @classmethod
    def from_file(cls, path: str | Path) -> ProtectedStore:
        path = Path(path)
        if not path.exists():
            return cls([])
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            [
                ProtectedSolution(
                    exercise_id=entry["exercise_id"],
                    title=entry.get("title", ""),
                    solution_text=entry.get("solution_text", ""),
                    answer_units=entry.get("answer_units", []),
                    page=entry.get("page"),
                )
                for entry in raw
            ]
        )

    # -- access ------------------------------------------------------------ #
    def get(self, exercise_id: str, accessor: str) -> ProtectedSolution | None:
        """Fetch a solution. `accessor` must name a safety component."""
        permitted = accessor in ALLOWED_ACCESSORS
        self.access_log.append(
            {"exercise_id": exercise_id, "accessor": accessor, "permitted": permitted}
        )
        if not permitted:
            raise ProtectedAccessViolation(
                f"component {accessor!r} may not read protected solutions; "
                f"permitted accessors: {sorted(ALLOWED_ACCESSORS)}"
            )
        return self._by_id.get(exercise_id)

    def has(self, exercise_id: str | None) -> bool:
        """Existence check. Reveals nothing about content, so it is unrestricted."""
        return bool(exercise_id) and exercise_id in self._by_id

    def unit_count(self, exercise_id: str) -> int:
        solution = self._by_id.get(exercise_id)
        return len(solution.answer_units) if solution else 0

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def violations(self) -> list[dict[str, Any]]:
        return [entry for entry in self.access_log if not entry["permitted"]]

    def __len__(self) -> int:
        return len(self._by_id)
