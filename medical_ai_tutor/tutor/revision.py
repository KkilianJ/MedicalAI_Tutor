"""Critique-guided revision.

When the safety gate returns REVISE, the response is regenerated with the
specific critique attached: which answer units leaked and why. The instruction is
never a generic "say it more safely" — the pedagogical action still has to
happen, minus the disclosure.

The loop is hard-bounded by MAX_REVISIONS_PER_TURN; when the budget runs out the
runtime emits the deterministic safe fallback instead.
"""

from __future__ import annotations

import json

from ..config import Config
from ..llm.base import LLMClient, LLMMessage
from ..llm.prompt_loader import load_prompt
from ..safety.gate import SafetyGate
from ..state.models import (
    ExerciseMeta,
    LearnerDiagnosis,
    LearnerState,
    PedagogicalDecision,
    RetrievedPassage,
    SafetyReport,
    SafetyVerdict,
)
from ..tracing.models import RevisionRecord
from .context import build_revision_context
from .fallback import safe_fallback_text


class RevisionLoop:
    """Bounded regenerate-and-recheck loop."""

    def __init__(self, client: LLMClient, config: Config, gate: SafetyGate) -> None:
        self.client = client
        self.gate = gate
        self.model = config.get("llm.models.generator", "claude-sonnet-5")
        self.max_tokens = int(config.get("llm.max_output_tokens", 1200))
        self.temperature = float(config.get("llm.temperature", 0.2))
        self.max_revisions = int(config.get("limits.max_revisions_per_turn", 2))
        self.dialogue_turns = int(config.get("context.recent_dialogue_turns", 6))
        self.max_passage_chars = int(config.get("context.max_passage_chars", 1100))
        self.max_previous_hints = int(config.get("context.max_previous_hints", 5))
        self.system = load_prompt("revision")

    def run(
        self,
        candidate: str,
        report: SafetyReport,
        state: LearnerState,
        diagnosis: LearnerDiagnosis,
        decision: PedagogicalDecision,
        passages: list[RetrievedPassage],
        exercise: ExerciseMeta | None,
    ) -> tuple[str, SafetyReport, list[RevisionRecord], bool, int, int]:
        """Revise until safe or out of budget.

        Returns (text, final_report, records, used_fallback, in_tokens, out_tokens).
        """
        records: list[RevisionRecord] = []
        input_tokens = output_tokens = 0
        current_text, current_report = candidate, report

        for attempt in range(1, self.max_revisions + 1):
            records.append(
                RevisionRecord(
                    attempt=attempt,
                    reason_code=current_report.reason_code,
                    leaked_units=current_report.leaked_units[:6],
                    candidate_preview=current_text[:160],
                )
            )

            context = build_revision_context(
                state=state,
                diagnosis=diagnosis,
                decision=decision,
                previous_candidate=current_text,
                leaked_units=current_report.leaked_units[:8],
                safety_reason=current_report.reason_code,
                passages=passages,
                exercise=exercise,
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
                revised = result.text.strip()
                input_tokens += result.input_tokens
                output_tokens += result.output_tokens
            except Exception:
                break  # provider failure: stop and take the deterministic fallback

            if not revised:
                break

            current_text = revised
            current_report = self.gate.evaluate(current_text, state, exercise)
            if current_report.verdict == SafetyVerdict.PASS:
                return current_text, current_report, records, False, input_tokens, output_tokens
            if current_report.verdict == SafetyVerdict.BLOCK:
                break  # a BLOCK is not worth another revision attempt

        # Budget exhausted, blocked, or the provider failed: emit the safe
        # fallback and re-verify it so nothing unchecked ever reaches the learner.
        fallback = safe_fallback_text(decision, state)
        fallback_report = self.gate.evaluate(fallback, state, exercise)
        return fallback, fallback_report, records, True, input_tokens, output_tokens
