"""Execution trace models.

A trace records every transition of one turn so the agent loop is inspectable
after the fact: what was observed, what was diagnosed, how state changed, which
action was chosen, which tools ran, and what safety decided.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from ..state.models import (
    DeterministicFinding,
    JudgeVerdict,
    LearnerDiagnosis,
    Observation,
    PedagogicalDecision,
    SafetyVerdict,
    TurnMetrics,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StageRecord(BaseModel):
    """One named stage of the turn pipeline."""

    name: str
    started_at: datetime = Field(default_factory=_utcnow)
    duration_ms: float = 0.0
    ok: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class StateDelta(BaseModel):
    """Auditable record of how the learner model changed this turn."""

    mastery_changes: list[dict[str, Any]] = Field(default_factory=list)
    misconceptions_added: list[str] = Field(default_factory=list)
    misconceptions_resolved: list[str] = Field(default_factory=list)
    hint_level_before: int = 0
    hint_level_after: int = 0
    attempt_count_before: int = 0
    attempt_count_after: int = 0
    topic_before: str | None = None
    topic_after: str | None = None
    exercise_before: str | None = None
    exercise_after: str | None = None


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    result_summary: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None


class RevisionRecord(BaseModel):
    attempt: int
    reason_code: str
    leaked_units: list[str] = Field(default_factory=list)
    candidate_preview: str = ""


class TurnTrace(BaseModel):
    """The complete, durable record of one `run_turn()` execution."""

    trace_id: str
    session_id: str
    turn_id: int
    created_at: datetime = Field(default_factory=_utcnow)

    student_message: str = ""
    observations: list[Observation] = Field(default_factory=list)

    diagnosis: LearnerDiagnosis | None = None
    state_delta: StateDelta | None = None
    decision: PedagogicalDecision | None = None

    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    retrieved_section_ids: list[str] = Field(default_factory=list)

    candidate_response: str = ""
    deterministic: DeterministicFinding | None = None
    judge: JudgeVerdict | None = None
    safety_verdict: SafetyVerdict = SafetyVerdict.PASS
    revisions: list[RevisionRecord] = Field(default_factory=list)
    used_fallback: bool = False
    final_response: str = ""

    stages: list[StageRecord] = Field(default_factory=list)
    metrics: TurnMetrics = Field(default_factory=TurnMetrics)

    # Stage names every complete turn must contain. Asserted by tests so the
    # trace cannot silently lose observability. ClassVar, so it is metadata about
    # the schema rather than a field stored in every trace row.
    REQUIRED_STAGES: ClassVar[tuple[str, ...]] = (
        "observe",
        "diagnose",
        "update_state",
        "decide",
        "tools",
        "generate",
        "safety",
        "persist",
    )

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def failed_stages(self) -> list[tuple[str, str]]:
        """(stage, error) for every stage that did not complete cleanly.

        The runtime degrades gracefully when a provider call fails — a neutral
        diagnosis, a default action, a deterministic fallback response. That is
        the right behaviour, but it must never be silent: a turn where all three
        model calls 404'd looks, to a reader, exactly like a turn where the tutor
        chose to ask a diagnostic question.
        """
        return [(s.name, s.error or "failed") for s in self.stages if not s.ok or s.error]

    def has_required_stages(self) -> bool:
        present = set(self.stage_names())
        return all(name in present for name in self.REQUIRED_STAGES)

    # ------------------------------------------------------------------ #
    # Redaction
    # ------------------------------------------------------------------ #
    def redacted(self) -> TurnTrace:
        """A copy safe to expose outside the safety subsystem.

        A trace legitimately records things no public surface may show:

        * `candidate_response` is the text *before* the safety gate ran, so on a
          flagged turn it may contain the disclosure that was caught;
        * `deterministic.matched_phrases` are the literal overlapping spans;
        * `revisions[].candidate_preview` is an excerpt of a flagged candidate.

        The API and the UI serve this view, never the raw trace.
        """
        copy = self.model_copy(deep=True)
        # The candidate is safe to show only when it is exactly what was emitted.
        # Any divergence — a revision, a block, a fallback — means the candidate
        # was rejected, and a rejected candidate is the one that may carry the
        # disclosure. The final verdict is not a usable signal here: the fallback
        # path re-verifies and always ends at PASS.
        flagged = (
            copy.candidate_response.strip() != copy.final_response.strip()
            or copy.safety_verdict != SafetyVerdict.PASS
            or bool(copy.revisions)
            or copy.used_fallback
        )

        if copy.deterministic is not None:
            copy.deterministic = copy.deterministic.model_copy(
                update={
                    "matched_phrases": [
                        f"[redacted: {len(p.split())} tokens]"
                        for p in copy.deterministic.matched_phrases
                    ]
                }
            )
        if flagged:
            copy.candidate_response = (
                f"[redacted: pre-safety candidate, {len(self.candidate_response)} chars]"
            )
        copy.revisions = [
            r.model_copy(update={"candidate_preview": "[redacted]"}) for r in copy.revisions
        ]
        # Tool observations can carry long passage text; keep traces compact.
        for observation in copy.observations:
            for key in ("passages", "text"):
                if key in observation.content and isinstance(observation.content[key], list):
                    observation.content[key] = f"[{len(observation.content[key])} items]"
        return copy
