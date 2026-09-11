"""The safety gate.

Combines the deterministic detector (Layer A) and the semantic judge (Layer B)
into one verdict for a candidate response.

The deterministic layer can only ever make the verdict *stricter*. A lenient
model opinion cannot overturn a measured verbatim overlap, which is what keeps
the protected-solution guarantee from resting on a prompt.
"""

from __future__ import annotations

from ..config import Config
from ..llm.base import LLMClient
from ..state.models import (
    DeterministicFinding,
    ExerciseMeta,
    JudgeVerdict,
    LearnerState,
    SafetyReport,
    SafetyVerdict,
)
from .deterministic import DeterministicDetector
from .protected_store import ProtectedStore
from .semantic_judge import SemanticJudge

_SEVERITY = {SafetyVerdict.PASS: 0, SafetyVerdict.REVISE: 1, SafetyVerdict.BLOCK: 2}


def most_severe(*verdicts: SafetyVerdict) -> SafetyVerdict:
    return max(verdicts, key=lambda v: _SEVERITY[v])


class SafetyGate:
    """Evaluates candidate tutor responses against protected solutions."""

    def __init__(
        self,
        config: Config,
        protected_store: ProtectedStore,
        client: LLMClient | None = None,
        judge_model: str | None = None,
    ) -> None:
        safety_config = config.section("safety")
        self.store = protected_store
        self.detector = DeterministicDetector.from_config(safety_config.get("deterministic", {}))
        self.high_hint_level = int(safety_config.get("high_hint_level", 3))

        judge_config = safety_config.get("judge", {})
        self.judge_enabled = bool(judge_config.get("enabled", True)) and client is not None
        self.judge: SemanticJudge | None = None
        if self.judge_enabled and client is not None:
            self.judge = SemanticJudge(
                client=client,
                model=judge_model or config.get("llm.models.judge", "claude-sonnet-5"),
                on_error_verdict=SafetyVerdict(
                    judge_config.get("on_error_verdict", "REVISE")
                ),
                max_repair_attempts=int(config.get("llm.max_schema_repair_attempts", 1)),
            )
        # Usage attributable to safety, surfaced in per-turn metrics.
        self.last_judge_tokens: tuple[int, int] = (0, 0)
        self.judge_calls = 0

    # ------------------------------------------------------------------ #
    def is_protected_active(self, state: LearnerState) -> bool:
        return self.store.has(state.current_exercise_id)

    def _deterministic_verdict(self, finding: DeterministicFinding) -> SafetyVerdict:
        if not finding.leaked:
            return SafetyVerdict.PASS
        severe = (
            len(finding.covered_units) >= 3
            or finding.unit_coverage >= 0.5
            or finding.max_verbatim_run >= 2 * self.detector.max_verbatim_run
        )
        return SafetyVerdict.BLOCK if severe else SafetyVerdict.REVISE

    def evaluate(
        self,
        candidate: str,
        state: LearnerState,
        exercise: ExerciseMeta | None = None,
    ) -> SafetyReport:
        """Evaluate one candidate response."""
        exercise_id = state.current_exercise_id
        if not self.store.has(exercise_id):
            # No protected exercise is active: nothing to leak.
            return SafetyReport(
                verdict=SafetyVerdict.PASS,
                protected_exercise_active=False,
                reason_code="no_protected_exercise",
            )

        solution = self.store.get(exercise_id, accessor="safety_gate")
        if solution is None:  # pragma: no cover - guarded by has()
            return SafetyReport(
                verdict=SafetyVerdict.PASS,
                protected_exercise_active=False,
                reason_code="solution_missing",
            )

        finding = self.detector.check(
            candidate=candidate,
            solution_text=solution.solution_text,
            answer_units=solution.answer_units,
            unit_labels=solution.unit_labels(),
        )
        deterministic_verdict = self._deterministic_verdict(finding)

        judge_verdict: JudgeVerdict | None = None
        if self.judge is not None:
            judge_verdict, input_tokens, output_tokens = self.judge.judge(
                candidate=candidate,
                solution=solution,
                exercise_question=(exercise.question if exercise else ""),
                hint_level=state.hint_level,
                deterministic=finding,
            )
            self.last_judge_tokens = (input_tokens, output_tokens)
            self.judge_calls += 1

        combined = deterministic_verdict
        if judge_verdict is not None:
            combined = most_severe(deterministic_verdict, judge_verdict.verdict)

        if combined == SafetyVerdict.PASS:
            reason_code = "clean"
        elif finding.reason_codes:
            reason_code = finding.reason_codes[0]
        elif judge_verdict is not None:
            reason_code = judge_verdict.reason_code
        else:
            reason_code = "unspecified"

        return SafetyReport(
            verdict=combined,
            deterministic=finding,
            judge=judge_verdict,
            protected_exercise_active=True,
            reason_code=reason_code,
        )

    def protected_unit_labels(self, exercise_id: str | None) -> list[str]:
        """Unit labels for the reviser. Labels only — never solution text."""
        if not self.store.has(exercise_id):
            return []
        solution = self.store.get(exercise_id, accessor="safety_gate")
        return solution.unit_labels() if solution else []
