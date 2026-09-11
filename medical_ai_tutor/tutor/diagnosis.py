"""Stage 1: learner diagnosis."""

from __future__ import annotations

import json

from ..config import Config
from ..llm.base import LLMClient, LLMMessage
from ..llm.prompt_loader import load_prompt
from ..state.models import (
    ExerciseMeta,
    Intent,
    LearnerDiagnosis,
    LearnerState,
    Observation,
    ResponseQuality,
)
from .context import build_diagnosis_context


class LearnerDiagnoser:
    """Turns a student message into a validated, structured diagnosis."""

    def __init__(
        self,
        client: LLMClient,
        config: Config,
        known_concepts: list[str] | None = None,
    ) -> None:
        self.client = client
        self.model = config.get("llm.models.controller", "claude-sonnet-5")
        self.max_tokens = int(config.get("llm.max_output_tokens", 1200))
        self.temperature = float(config.get("llm.temperature", 0.2))
        self.max_repair = int(config.get("llm.max_schema_repair_attempts", 1))
        self.dialogue_turns = int(config.get("context.recent_dialogue_turns", 6))
        self.known_concepts = known_concepts or []
        self.system = load_prompt("diagnosis")

    def run(
        self,
        observation: Observation,
        state: LearnerState,
        exercise: ExerciseMeta | None = None,
        protected: bool = False,
    ) -> tuple[LearnerDiagnosis, int, int, str | None]:
        """Returns (diagnosis, input_tokens, output_tokens, error)."""
        student_message = str(observation.content.get("message", ""))
        context = build_diagnosis_context(
            student_message=student_message,
            state=state,
            exercise=exercise,
            protected=protected,
            dialogue_turns=self.dialogue_turns,
            known_concepts=self.known_concepts,
        )
        messages = [LLMMessage(role="user", content=json.dumps(context, ensure_ascii=False))]

        try:
            diagnosis, result = self.client.structured(
                system=self.system,
                messages=messages,
                schema=LearnerDiagnosis,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_repair_attempts=self.max_repair,
            )
        except Exception as exc:
            # A failed diagnosis must not stall the turn; fall back to a neutral
            # one that keeps the learner model unchanged.
            return self._fallback(state), 0, 0, f"{type(exc).__name__}: {exc}"

        # The LLM may not silently move the learner to a different exercise than
        # the one the runtime knows about.
        if diagnosis.current_exercise_id and exercise is not None:
            if diagnosis.current_exercise_id != exercise.exercise_id:
                diagnosis = diagnosis.model_copy(
                    update={"current_exercise_id": exercise.exercise_id}
                )
        return diagnosis, result.input_tokens, result.output_tokens, None

    @staticmethod
    def _fallback(state: LearnerState) -> LearnerDiagnosis:
        return LearnerDiagnosis(
            intent=Intent.UNKNOWN,
            response_quality=ResponseQuality.NOT_APPLICABLE,
            current_topic=state.current_topic,
            current_exercise_id=state.current_exercise_id,
            confidence=0.0,
        )
