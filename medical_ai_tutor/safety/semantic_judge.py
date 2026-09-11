"""Layer B: independent semantic safety judge.

A separate LLM call with a separate system role. It is the only model call that
is shown the protected solution, and it is structurally unable to influence the
tutoring text: it returns a verdict, never prose for the student.
"""

from __future__ import annotations

import json

from ..llm.base import LLMClient, LLMMessage
from ..llm.prompt_loader import load_prompt
from ..state.models import DeterministicFinding, JudgeVerdict, SafetyVerdict
from .protected_store import ProtectedSolution


class SemanticJudge:
    """Judges whether a candidate response semantically discloses a solution."""

    def __init__(
        self,
        client: LLMClient,
        model: str,
        on_error_verdict: SafetyVerdict = SafetyVerdict.REVISE,
        max_tokens: int = 600,
        max_repair_attempts: int = 1,
    ) -> None:
        self.client = client
        self.model = model
        self.on_error_verdict = on_error_verdict
        self.max_tokens = max_tokens
        self.max_repair_attempts = max_repair_attempts
        self.system = load_prompt("safety_judge")
        self.last_error: str | None = None

    def judge(
        self,
        candidate: str,
        solution: ProtectedSolution,
        exercise_question: str,
        hint_level: int,
        deterministic: DeterministicFinding,
    ) -> tuple[JudgeVerdict, int, int]:
        """Returns (verdict, input_tokens, output_tokens)."""
        context = {
            "exercise_id": solution.exercise_id,
            "exercise_question": exercise_question,
            "protected_solution": solution.solution_text,
            "protected_answer_units": solution.answer_units,
            "protected_unit_labels": solution.unit_labels(),
            "candidate_response": candidate,
            "hint_level": hint_level,
            "deterministic_findings": deterministic.model_dump(mode="json"),
        }
        messages = [LLMMessage(role="user", content=json.dumps(context, ensure_ascii=False))]

        try:
            verdict, result = self.client.structured(
                system=self.system,
                messages=messages,
                schema=JudgeVerdict,
                model=self.model,
                temperature=0.0,
                max_tokens=self.max_tokens,
                max_repair_attempts=self.max_repair_attempts,
            )
        except Exception:
            # A judge that cannot answer must not become an implicit PASS.
            return (
                JudgeVerdict(
                    verdict=self.on_error_verdict,
                    reason_code="judge_unavailable",
                    confidence=0.0,
                    leaked_units=list(deterministic.covered_units),
                ),
                0,
                0,
            )

        verdict = self._sanitize(verdict, solution)
        return verdict, result.input_tokens, result.output_tokens

    @staticmethod
    def _sanitize(verdict: JudgeVerdict, solution: ProtectedSolution) -> JudgeVerdict:
        """Strip any solution text the judge may have echoed into `leaked_units`.

        The judge is told to emit labels only; this makes that a guarantee rather
        than an instruction, so a verdict can be safely written to a trace.
        """
        safe_units: list[str] = []
        for unit in verdict.leaked_units[:8]:
            text = str(unit)
            if len(text.split()) > 12:
                text = " ".join(text.split()[:6]) + "…"
            safe_units.append(text)
        return verdict.model_copy(update={"leaked_units": safe_units})
