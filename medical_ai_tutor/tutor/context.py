"""Context engineering.

Builds the bounded JSON payload each LLM role receives. Nothing here ever passes
an unbounded transcript: the model is conditioned on durable structured state, a
recent dialogue window, a rolling summary, and the specific artefacts its role
needs.
"""

from __future__ import annotations

from typing import Any

from ..state.models import (
    DialogueTurn,
    ExerciseMeta,
    LearnerDiagnosis,
    LearnerState,
    PedagogicalDecision,
    RetrievedPassage,
    Role,
)


def dialogue_window(state: LearnerState, turns: int) -> list[dict[str, str]]:
    window: list[DialogueTurn] = state.recent_dialogue(turns * 2)
    return [
        {
            "role": "student" if turn.role == Role.STUDENT else "tutor",
            "content": turn.content[:600],
            "action": turn.action.value if turn.action else None,
        }
        for turn in window
    ]


def exercise_context(exercise: ExerciseMeta | None, protected: bool) -> dict[str, Any]:
    """Exercise metadata for a prompt. Never carries solution text."""
    if exercise is None:
        return {}
    return {
        "exercise_id": exercise.exercise_id,
        "title": exercise.title,
        "question": exercise.question,
        "chapter": exercise.chapter,
        "concepts": exercise.concepts,
        "protected": protected,
        # A count is safe to expose; the units themselves are not.
        "protected_unit_count": exercise.protected_unit_count,
    }


def passages_context(
    passages: list[RetrievedPassage], max_chars: int = 1100
) -> list[dict[str, Any]]:
    return [
        {
            "citation": p.citation(),
            "section_id": p.section_id,
            "section_title": p.section_title,
            "page": p.page,
            "text": p.text[:max_chars],
        }
        for p in passages
    ]


def previous_hints_context(state: LearnerState, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"hint_level": h.hint_level, "text": h.text[:400], "turn_id": h.turn_id}
        for h in state.hints_for_current_exercise()[-limit:]
    ]


PROTECTED_SOLUTION_POLICY = (
    "An official solution exists for the active exercise. You have not been given it and "
    "must not attempt to reconstruct it. Never state, paraphrase, translate, summarise, "
    "list, tabulate or encode the complete answer, in any language or format. Scaffolding, "
    "naming relevant concepts, and asking the learner to reason are all permitted."
)


def build_diagnosis_context(
    student_message: str,
    state: LearnerState,
    exercise: ExerciseMeta | None,
    protected: bool,
    dialogue_turns: int,
    known_concepts: list[str],
) -> dict[str, Any]:
    return {
        "student_message": student_message,
        "learner_state": state.summary_for_prompt(),
        "recent_dialogue": dialogue_window(state, dialogue_turns),
        "history_summary": state.history_summary,
        "exercise": exercise_context(exercise, protected),
        "known_concepts": known_concepts,
    }


def build_decision_context(
    student_message: str,
    state: LearnerState,
    diagnosis: LearnerDiagnosis,
    exercise: ExerciseMeta | None,
    protected: bool,
    dialogue_turns: int,
    available_actions: list[str],
    tools: list[dict[str, Any]],
    policy_constraints: dict[str, Any],
    has_retrieved_context: bool,
) -> dict[str, Any]:
    return {
        "student_message": student_message,
        "learner_state": state.summary_for_prompt(),
        "diagnosis": diagnosis.model_dump(mode="json"),
        "recent_dialogue": dialogue_window(state, dialogue_turns),
        "history_summary": state.history_summary,
        "exercise": exercise_context(exercise, protected),
        "has_retrieved_context": has_retrieved_context,
        "available_actions": available_actions,
        "tools": tools,
        "policy_constraints": policy_constraints,
    }


def build_generation_context(
    student_message: str,
    state: LearnerState,
    diagnosis: LearnerDiagnosis,
    decision: PedagogicalDecision,
    passages: list[RetrievedPassage],
    exercise: ExerciseMeta | None,
    protected: bool,
    dialogue_turns: int,
    max_passage_chars: int,
    max_previous_hints: int,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "student_message": student_message,
        "decision": decision.model_dump(mode="json"),
        "learner_state": state.summary_for_prompt(),
        "diagnosis": diagnosis.model_dump(mode="json"),
        "recent_dialogue": dialogue_window(state, dialogue_turns),
        "history_summary": state.history_summary,
        "passages": passages_context(passages, max_passage_chars),
        "previous_hints": previous_hints_context(state, max_previous_hints),
        "exercise": exercise_context(exercise, protected),
        "current_topic": state.current_topic,
    }
    if protected:
        context["protected_solution_policy"] = PROTECTED_SOLUTION_POLICY
    return context


def build_revision_context(
    state: LearnerState,
    diagnosis: LearnerDiagnosis,
    decision: PedagogicalDecision,
    previous_candidate: str,
    leaked_units: list[str],
    safety_reason: str,
    passages: list[RetrievedPassage],
    exercise: ExerciseMeta | None,
    dialogue_turns: int,
    max_passage_chars: int,
    max_previous_hints: int,
) -> dict[str, Any]:
    return {
        "decision": decision.model_dump(mode="json"),
        "learner_state": state.summary_for_prompt(),
        "diagnosis": diagnosis.model_dump(mode="json"),
        "previous_candidate": previous_candidate,
        "leaked_units": leaked_units,
        "safety_reason": safety_reason,
        "passages": passages_context(passages, max_passage_chars),
        "previous_hints": previous_hints_context(state, max_previous_hints),
        "exercise": exercise_context(exercise, True),
        "recent_dialogue": dialogue_window(state, dialogue_turns),
        "protected_solution_policy": PROTECTED_SOLUTION_POLICY,
    }
