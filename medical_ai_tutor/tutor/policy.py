"""Stage 2: pedagogical policy.

The LLM selects one action; this module decides whether that selection is
allowed to stand. Two things are enforced in Python rather than in a prompt:

* the hint level is clamped, so scaffolding cannot be skipped to hand over an
  answer;
* a request for the official solution to a protected exercise is forced to
  `REFUSE_SOLUTION`, whatever the model proposed.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import Config
from ..llm.base import LLMClient, LLMMessage
from ..llm.prompt_loader import load_prompt
from ..state.models import (
    ALL_ACTIONS,
    ExerciseMeta,
    Intent,
    LearnerDiagnosis,
    LearnerState,
    PedagogicalAction,
    PedagogicalDecision,
)
from .context import build_decision_context

# Actions that hand the learner information rather than asking them to reason.
_INFORMATION_GIVING = {
    PedagogicalAction.EXPLAIN_CONCEPT,
    PedagogicalAction.GIVE_EXAMPLE,
    PedagogicalAction.GIVE_HINT,
}


class PedagogicalPolicy:
    """Selects and validates the single action taken this turn."""

    def __init__(self, client: LLMClient, config: Config) -> None:
        self.client = client
        self.model = config.get("llm.models.controller", "claude-sonnet-5")
        self.max_tokens = int(config.get("llm.max_output_tokens", 1200))
        self.temperature = float(config.get("llm.temperature", 0.2))
        self.max_repair = int(config.get("llm.max_schema_repair_attempts", 1))
        self.dialogue_turns = int(config.get("context.recent_dialogue_turns", 6))
        self.hint_min = int(config.get("pedagogy.hint_level_min", 0))
        self.hint_max = int(config.get("pedagogy.hint_level_max", 4))
        self.system = load_prompt("pedagogical_decision")
        self.override_count = 0

    def policy_constraints(self, state: LearnerState, protected: bool) -> dict[str, Any]:
        return {
            "objective": "Use the minimum assistance necessary to move the learner forward.",
            "hint_level_current": state.hint_level,
            "hint_level_max": self.hint_max,
            "attempt_count": state.attempt_count,
            "protected_exercise_active": protected,
            "protected_solution_rule": (
                "Never select an action whose purpose is to deliver the official solution."
            ),
            "retrieval_rule": (
                "Retrieve only when the answer depends on what this specific textbook says."
            ),
        }

    def select(
        self,
        state: LearnerState,
        diagnosis: LearnerDiagnosis,
        student_message: str,
        exercise: ExerciseMeta | None,
        protected: bool,
        tool_specs: list[dict[str, Any]],
        has_retrieved_context: bool = False,
    ) -> tuple[PedagogicalDecision, int, int, str | None]:
        """Returns (decision, input_tokens, output_tokens, error)."""
        context = build_decision_context(
            student_message=student_message,
            state=state,
            diagnosis=diagnosis,
            exercise=exercise,
            protected=protected,
            dialogue_turns=self.dialogue_turns,
            available_actions=[a.value for a in ALL_ACTIONS],
            tools=tool_specs,
            policy_constraints=self.policy_constraints(state, protected),
            has_retrieved_context=has_retrieved_context,
        )
        messages = [LLMMessage(role="user", content=json.dumps(context, ensure_ascii=False))]

        error: str | None = None
        try:
            decision, result = self.client.structured(
                system=self.system,
                messages=messages,
                schema=PedagogicalDecision,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_repair_attempts=self.max_repair,
            )
            input_tokens, output_tokens = result.input_tokens, result.output_tokens
        except Exception as exc:
            decision = self._fallback(state, diagnosis)
            input_tokens = output_tokens = 0
            error = f"{type(exc).__name__}: {exc}"

        return self.enforce(decision, state, diagnosis, protected, has_retrieved_context), (
            input_tokens
        ), output_tokens, error

    # ------------------------------------------------------------------ #
    def enforce(
        self,
        decision: PedagogicalDecision,
        state: LearnerState,
        diagnosis: LearnerDiagnosis,
        protected: bool,
        has_retrieved_context: bool,
    ) -> PedagogicalDecision:
        """Apply the constraints the runtime owns."""
        updates: dict[str, Any] = {}

        # 1. Hint level is bounded and may advance by at most one step per turn.
        desired = max(self.hint_min, min(self.hint_max, decision.desired_hint_level))
        if decision.action == PedagogicalAction.GIVE_HINT:
            desired = max(self.hint_min, min(self.hint_max, min(desired, state.hint_level + 1)))
            if desired <= state.hint_level:
                desired = min(self.hint_max, state.hint_level + 1)
        if desired != decision.desired_hint_level:
            updates["desired_hint_level"] = desired

        # 2. Asking for a protected solution can only ever produce a refusal.
        if protected and diagnosis.intent == Intent.DIRECT_ANSWER_REQUEST:
            if decision.action != PedagogicalAction.REFUSE_SOLUTION:
                self.override_count += 1
                updates["action"] = PedagogicalAction.REFUSE_SOLUTION
                updates["reason_code"] = "runtime_override_protected_solution"
                updates["needs_retrieval"] = False

        # 3. A prompt-injection attempt is redirected, never obeyed.
        if diagnosis.intent == Intent.PROMPT_INJECTION and decision.action not in (
            PedagogicalAction.REDIRECT,
            PedagogicalAction.REFUSE_SOLUTION,
        ):
            self.override_count += 1
            updates["action"] = PedagogicalAction.REDIRECT
            updates["reason_code"] = "runtime_override_injection"
            updates["needs_retrieval"] = False

        # 4. Do not retrieve twice for the same turn.
        if has_retrieved_context and decision.needs_retrieval:
            updates["needs_retrieval"] = False
            if decision.action == PedagogicalAction.RETRIEVE:
                updates["action"] = PedagogicalAction.EXPLAIN_CONCEPT
                updates["reason_code"] = "retrieval_already_available"

        # 5. Target concept defaults to the current topic so the generator always
        #    has something concrete to aim at.
        if not decision.target_concept:
            updates["target_concept"] = diagnosis.current_topic or state.current_topic

        return decision.model_copy(update=updates) if updates else decision

    @staticmethod
    def _fallback(state: LearnerState, diagnosis: LearnerDiagnosis) -> PedagogicalDecision:
        """Deterministic decision used when the policy LLM is unavailable.

        Asking the learner a diagnostic question is the safest possible default:
        it reveals nothing and still moves the conversation forward.
        """
        return PedagogicalDecision(
            action=PedagogicalAction.ASK_DIAGNOSTIC,
            target_concept=diagnosis.current_topic or state.current_topic,
            needs_retrieval=False,
            desired_hint_level=state.hint_level,
            reason_code="policy_unavailable_fallback",
            confidence=0.0,
        )

    @staticmethod
    def is_information_giving(action: PedagogicalAction) -> bool:
        return action in _INFORMATION_GIVING
