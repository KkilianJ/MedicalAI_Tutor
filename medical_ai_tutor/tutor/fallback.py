"""Deterministic safe fallback.

Used when generation fails, or when revision has exhausted its budget and the
candidate is still unsafe. It contains no model call and no protected content,
so it is always available and always safe.

A fallback is not an apology: it still performs a useful pedagogical move.
"""

from __future__ import annotations

from ..state.models import LearnerState, PedagogicalAction, PedagogicalDecision

_BY_ACTION: dict[PedagogicalAction, str] = {
    PedagogicalAction.ASK_DIAGNOSTIC: (
        "Before we go further, tell me in your own words what you think {concept} is for. "
        "Even a rough answer tells me where to start."
    ),
    PedagogicalAction.ASK_SOCRATIC: (
        "Let us take one step at a time. What problem do you think {concept} is meant to solve, "
        "and who would notice if it were missing?"
    ),
    PedagogicalAction.GIVE_HINT: (
        "Rather than another hint, try this: write down the single property of {concept} that the "
        "exercise depends on. Which part of that property are you unsure about?"
    ),
    PedagogicalAction.GIVE_EXAMPLE: (
        "Think of a concrete case from a hospital you can picture, and describe how {concept} "
        "would show up in it. We can work from your example."
    ),
    PedagogicalAction.EXPLAIN_CONCEPT: (
        "Let us build this up from what you already know. Which part of {concept} is clear to you, "
        "and where does it stop making sense?"
    ),
    PedagogicalAction.CORRECT_MISCONCEPTION: (
        "There is one point in your reasoning about {concept} worth testing. Describe a situation "
        "where your explanation would fail — what would go wrong there?"
    ),
    PedagogicalAction.CHALLENGE: (
        "Try applying {concept} to a different setting than the one we discussed. What changes, "
        "and what stays the same?"
    ),
    PedagogicalAction.QUIZ: (
        "Quick check on {concept}: state one property it guarantees, and one situation where that "
        "guarantee is hard to keep."
    ),
    PedagogicalAction.RETRIEVE: (
        "Let us anchor this in the course material. Which aspect of {concept} should we look up "
        "first?"
    ),
    PedagogicalAction.REDIRECT: (
        "Let us stay with the course material. Which part of the health information systems topic "
        "would you like to work through next?"
    ),
    PedagogicalAction.REFUSE_SOLUTION: (
        "I am not going to hand over the official answer — producing it is the exercise. "
        "Let us take the first step instead: which single thing does the question actually ask "
        "you to identify about {concept}?"
    ),
}

_GENERIC = (
    "Let us take this one step at a time. Describe what you have worked out so far, and where "
    "your reasoning stops."
)


def safe_fallback_text(
    decision: PedagogicalDecision | None = None,
    state: LearnerState | None = None,
) -> str:
    """A safe, useful response requiring no model call and no protected data."""
    if decision is None:
        return _GENERIC
    concept = decision.target_concept or (state.current_topic if state else None) or "this topic"
    template = _BY_ACTION.get(decision.action, _GENERIC)
    return template.format(concept=str(concept).replace("_", " "))
