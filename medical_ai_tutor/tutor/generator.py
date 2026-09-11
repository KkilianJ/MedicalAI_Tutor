"""Stage 3: tutor generation.

Generation is deliberately separate from decision-making. The generator is given
a decision it must execute and the safe material it may use; it does not get to
reconsider the pedagogy, and it is never given protected solution text.
"""

from __future__ import annotations

import json

from ..config import Config
from ..llm.base import LLMClient, LLMMessage
from ..llm.prompt_loader import load_prompt
from ..state.models import (
    ExerciseMeta,
    LearnerDiagnosis,
    LearnerState,
    PedagogicalAction,
    PedagogicalDecision,
    RetrievedPassage,
)
from .context import build_generation_context

# Coarse labels recorded in learner state describing what a turn disclosed.
# They describe the *kind* of disclosure, never its content.
_DISCLOSURE_LABELS = {
    PedagogicalAction.GIVE_HINT: "hint",
    PedagogicalAction.GIVE_EXAMPLE: "worked_example",
    PedagogicalAction.EXPLAIN_CONCEPT: "concept_explanation",
    PedagogicalAction.CORRECT_MISCONCEPTION: "misconception_correction",
}


class TutorGenerator:
    """Produces the student-facing message."""

    def __init__(self, client: LLMClient, config: Config) -> None:
        self.client = client
        self.model = config.get("llm.models.generator", "claude-sonnet-5")
        self.max_tokens = int(config.get("llm.max_output_tokens", 1200))
        self.temperature = float(config.get("llm.generator_temperature", 0.5))
        self.dialogue_turns = int(config.get("context.recent_dialogue_turns", 6))
        self.max_passage_chars = int(config.get("context.max_passage_chars", 1100))
        self.max_previous_hints = int(config.get("context.max_previous_hints", 5))
        self.system = load_prompt("tutor_generation")

    def generate(
        self,
        state: LearnerState,
        diagnosis: LearnerDiagnosis,
        decision: PedagogicalDecision,
        student_message: str,
        passages: list[RetrievedPassage],
        exercise: ExerciseMeta | None,
        protected: bool,
    ) -> tuple[str, int, int, str | None]:
        """Returns (candidate_text, input_tokens, output_tokens, error)."""
        context = build_generation_context(
            student_message=student_message,
            state=state,
            diagnosis=diagnosis,
            decision=decision,
            passages=passages,
            exercise=exercise,
            protected=protected,
            dialogue_turns=self.dialogue_turns,
            max_passage_chars=self.max_passage_chars,
            max_previous_hints=self.max_previous_hints,
        )
        messages = [LLMMessage(role="user", content=json.dumps(context, ensure_ascii=False))]

        try:
            result = self.client.complete(
                system=self.system,
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            return "", 0, 0, f"{type(exc).__name__}: {exc}"

        return result.text.strip(), result.input_tokens, result.output_tokens, None

    @staticmethod
    def disclosure_units(decision: PedagogicalDecision) -> list[str]:
        """Coarse labels of what this action revealed, for the learner state."""
        label = _DISCLOSURE_LABELS.get(decision.action)
        if not label:
            return []
        concept = decision.target_concept or "topic"
        if decision.action == PedagogicalAction.GIVE_HINT:
            return [f"{label}:{concept}:L{decision.desired_hint_level}"]
        return [f"{label}:{concept}"]
