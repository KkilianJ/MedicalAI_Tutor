"""Deterministic learner-state transitions.

The diagnosis LLM supplies *evidence*; this module owns the arithmetic. That
split is deliberate: an LLM may not set mastery to an arbitrary number, and every
change it does cause is bounded, reproducible, and recorded in a `StateDelta`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..config import Config
from ..tracing.models import StateDelta
from .models import (
    ConceptMastery,
    DialogueTurn,
    HintRecord,
    LearnerDiagnosis,
    LearnerState,
    MasteryEvidence,
    Misconception,
    Observation,
    PedagogicalAction,
    PedagogicalDecision,
    ResponseQuality,
    Role,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class StateUpdater:
    """Applies validated diagnoses and tutor actions to a `LearnerState`."""

    def __init__(self, config: Config) -> None:
        ped = config.section("pedagogy")
        mastery = ped.get("mastery", {})
        self.correct_gain: float = mastery.get("correct_gain", 0.18)
        self.partial_gain: float = mastery.get("partial_gain", 0.09)
        self.incorrect_penalty: float = mastery.get("incorrect_penalty", 0.07)
        self.unclear_penalty: float = mastery.get("unclear_penalty", 0.0)
        self.decay: float = mastery.get("diminishing_decay", 0.35)
        self.confidence_gain: float = mastery.get("confidence_gain", 0.12)
        self.floor: float = mastery.get("mastery_floor", 0.0)
        self.ceiling: float = mastery.get("mastery_ceiling", 1.0)

        mis = ped.get("misconception", {})
        self.mis_initial: float = mis.get("initial_confidence", 0.6)
        self.mis_reinforce: float = mis.get("reinforce_gain", 0.15)
        self.mis_resolve_decay: float = mis.get("resolve_decay", 0.45)
        self.mis_deactivate_below: float = mis.get("deactivate_below", 0.25)

        self.demonstrated_threshold: float = ped.get("demonstrated_threshold", 0.7)
        band = ped.get("uncertain_band", [0.25, 0.7])
        self.uncertain_low: float = float(band[0])
        self.uncertain_high: float = float(band[1])

        self.hint_min: int = ped.get("hint_level_min", 0)
        self.hint_max: int = ped.get("hint_level_max", 4)

        ctx = config.section("context")
        self.max_dialogue: int = int(ctx.get("recent_dialogue_turns", 6)) * 4
        self.summary_max_chars: int = int(ctx.get("summary_max_chars", 900))
        self.max_previous_hints: int = int(ctx.get("max_previous_hints", 5))

    # ------------------------------------------------------------------ #
    # Mastery
    # ------------------------------------------------------------------ #
    def _raw_delta(self, quality: ResponseQuality) -> float:
        """Signed base delta for a quality label.

        Incorrect evidence lowers the estimate but never zeroes it: one wrong
        answer is weak evidence of total non-understanding.
        """
        return {
            ResponseQuality.CORRECT: self.correct_gain,
            ResponseQuality.PARTIALLY_CORRECT: self.partial_gain,
            ResponseQuality.INCORRECT: -self.incorrect_penalty,
            ResponseQuality.UNCLEAR: -self.unclear_penalty,
            ResponseQuality.NOT_APPLICABLE: 0.0,
        }.get(quality, 0.0)

    def _apply_evidence(
        self, state: LearnerState, evidence: MasteryEvidence, turn_id: int
    ) -> dict[str, float | str | int] | None:
        entry = state.concept_mastery.get(evidence.concept_id)
        if entry is None:
            entry = ConceptMastery(concept_id=evidence.concept_id, score=0.0, confidence=0.0)

        base = self._raw_delta(evidence.quality) * _clamp(evidence.weight)
        if base == 0.0 and evidence.quality != ResponseQuality.NOT_APPLICABLE:
            return None

        # Repeated evidence about the same concept has diminishing impact.
        damped = base / (1.0 + self.decay * entry.evidence_count)
        before = entry.score
        after = _clamp(before + damped, self.floor, self.ceiling)

        updated = ConceptMastery(
            concept_id=entry.concept_id,
            score=after,
            confidence=_clamp(entry.confidence + self.confidence_gain * _clamp(evidence.weight)),
            evidence_count=entry.evidence_count + 1,
            last_updated_turn=turn_id,
        )
        state.concept_mastery[evidence.concept_id] = updated

        return {
            "concept_id": evidence.concept_id,
            "quality": evidence.quality.value,
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(after - before, 4),
            "evidence_count": updated.evidence_count,
        }

    def _recompute_buckets(self, state: LearnerState) -> None:
        demonstrated, uncertain = [], []
        for cid, cm in state.concept_mastery.items():
            if cm.score >= self.demonstrated_threshold:
                demonstrated.append(cid)
            elif self.uncertain_low <= cm.score < self.uncertain_high:
                uncertain.append(cid)
        state.demonstrated_concepts = sorted(demonstrated)
        state.uncertain_concepts = sorted(uncertain)

    # ------------------------------------------------------------------ #
    # Misconceptions
    # ------------------------------------------------------------------ #
    def _apply_misconceptions(
        self, state: LearnerState, diagnosis: LearnerDiagnosis, turn_id: int
    ) -> tuple[list[str], list[str]]:
        added: list[str] = []
        resolved: list[str] = []

        for detected in diagnosis.detected_misconceptions:
            existing = state.find_misconception(detected.concept_id)
            if existing is None:
                state.misconceptions.append(
                    Misconception(
                        concept_id=detected.concept_id,
                        description=detected.description,
                        confidence=_clamp(detected.confidence or self.mis_initial),
                        active=True,
                        first_seen_turn=turn_id,
                        last_seen_turn=turn_id,
                    )
                )
                added.append(detected.concept_id)
            else:
                # Seeing it again reinforces belief that it is real.
                existing.confidence = _clamp(existing.confidence + self.mis_reinforce)
                existing.active = True
                existing.description = detected.description or existing.description
                existing.last_seen_turn = turn_id
                added.append(detected.concept_id)

        for concept_id in diagnosis.resolved_misconceptions:
            existing = state.find_misconception(concept_id)
            if existing is None or not existing.active:
                continue
            # Resolution lowers confidence; history is kept, never deleted.
            existing.confidence = _clamp(existing.confidence - self.mis_resolve_decay)
            existing.last_seen_turn = turn_id
            if existing.confidence < self.mis_deactivate_below:
                existing.active = False
                resolved.append(concept_id)

        return added, resolved

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def apply(
        self,
        state: LearnerState,
        diagnosis: LearnerDiagnosis,
        observation: Observation,
    ) -> tuple[LearnerState, StateDelta]:
        """Apply a validated diagnosis. Returns the new state and an audit delta."""
        new_state = state.model_copy(deep=True)
        turn_id = observation.turn_id

        delta = StateDelta(
            hint_level_before=new_state.hint_level,
            hint_level_after=new_state.hint_level,
            attempt_count_before=new_state.attempt_count,
            attempt_count_after=new_state.attempt_count,
            topic_before=new_state.current_topic,
            exercise_before=new_state.current_exercise_id,
        )

        for evidence in diagnosis.mastery_evidence:
            change = self._apply_evidence(new_state, evidence, turn_id)
            if change:
                delta.mastery_changes.append(change)

        added, resolved = self._apply_misconceptions(new_state, diagnosis, turn_id)
        delta.misconceptions_added = added
        delta.misconceptions_resolved = resolved

        if diagnosis.current_topic:
            new_state.current_topic = diagnosis.current_topic
        if diagnosis.current_exercise_id:
            if diagnosis.current_exercise_id != new_state.current_exercise_id:
                # Switching exercises resets scaffolding for the new problem.
                new_state.hint_level = self.hint_min
                new_state.attempt_count = 0
                new_state.revealed_information_units = []
            new_state.current_exercise_id = diagnosis.current_exercise_id

        from .models import Intent  # local import avoids a cycle at module import

        if diagnosis.intent == Intent.STUDENT_ATTEMPT or (
            diagnosis.intent == Intent.ANSWER_TO_TUTOR_QUESTION
            and diagnosis.response_quality != ResponseQuality.NOT_APPLICABLE
        ):
            new_state.attempt_count += 1

        new_state.latest_intent = diagnosis.intent
        new_state.latest_response_quality = diagnosis.response_quality
        new_state.turn_count = max(new_state.turn_count, turn_id)

        self._recompute_buckets(new_state)

        new_state.dialogue.append(
            DialogueTurn(
                role=Role.STUDENT,
                content=str(observation.content.get("message", "")),
                turn_id=turn_id,
            )
        )
        self._trim_dialogue(new_state)

        delta.hint_level_after = new_state.hint_level
        delta.attempt_count_after = new_state.attempt_count
        delta.topic_after = new_state.current_topic
        delta.exercise_after = new_state.current_exercise_id
        new_state.updated_at = datetime.now(UTC)
        return new_state, delta

    def record_action_and_response(
        self,
        state: LearnerState,
        decision: PedagogicalDecision,
        response_text: str,
        turn_id: int,
        delta: StateDelta | None = None,
        revealed_units: list[str] | None = None,
    ) -> LearnerState:
        """Record what the tutor actually did, after safety cleared it."""
        new_state = state.model_copy(deep=True)
        new_state.previous_actions.append(decision.action)
        new_state.previous_actions = new_state.previous_actions[-20:]

        if decision.action == PedagogicalAction.GIVE_HINT:
            # Hint level advances only when a hint is really delivered, and is
            # hard-bounded regardless of what the policy asked for.
            target = max(new_state.hint_level + 1, decision.desired_hint_level)
            new_state.hint_level = max(self.hint_min, min(self.hint_max, target))
            new_state.previous_hints.append(
                HintRecord(
                    exercise_id=new_state.current_exercise_id,
                    hint_level=new_state.hint_level,
                    text=response_text[:600],
                    turn_id=turn_id,
                )
            )
            new_state.previous_hints = new_state.previous_hints[-self.max_previous_hints :]

        if revealed_units:
            merged = list(new_state.revealed_information_units)
            for unit in revealed_units:
                if unit not in merged:
                    merged.append(unit)
            new_state.revealed_information_units = merged[:20]

        new_state.dialogue.append(
            DialogueTurn(
                role=Role.TUTOR,
                content=response_text,
                turn_id=turn_id,
                action=decision.action,
            )
        )
        self._trim_dialogue(new_state)
        new_state.turn_count = max(new_state.turn_count, turn_id)
        new_state.updated_at = datetime.now(UTC)

        if delta is not None:
            delta.hint_level_after = new_state.hint_level
        return new_state

    # ------------------------------------------------------------------ #
    # Context engineering
    # ------------------------------------------------------------------ #
    def _trim_dialogue(self, state: LearnerState) -> None:
        """Keep a bounded verbatim window; fold older turns into a summary.

        This is the whole of the system's "memory" policy: durable structured
        state plus a recent window plus a short rolling summary. No unbounded
        transcript is ever sent to a model.
        """
        if len(state.dialogue) <= self.max_dialogue:
            return
        overflow = state.dialogue[: len(state.dialogue) - self.max_dialogue]
        state.dialogue = state.dialogue[len(state.dialogue) - self.max_dialogue :]

        fragments = [state.history_summary] if state.history_summary else []
        for turn in overflow:
            label = "S" if turn.role == Role.STUDENT else "T"
            fragments.append(f"{label}{turn.turn_id}: {turn.content.strip()[:140]}")
        summary = " | ".join(f for f in fragments if f)
        if len(summary) > self.summary_max_chars:
            summary = "…" + summary[-(self.summary_max_chars - 1) :]
        state.history_summary = summary
