"""Schema validation is a runtime guarantee, not a convention."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medical_ai_tutor.state.models import (
    ConceptMastery,
    LearnerState,
    Misconception,
    PedagogicalAction,
    PedagogicalDecision,
)


def test_invalid_action_is_rejected():
    with pytest.raises(ValidationError):
        PedagogicalDecision(action="GIVE_THE_ANSWER")


@pytest.mark.parametrize("score", [-0.1, 1.1, 42.0])
def test_mastery_score_must_be_a_probability(score):
    with pytest.raises(ValidationError):
        ConceptMastery(concept_id="c", score=score)


@pytest.mark.parametrize("confidence", [-0.01, 1.5])
def test_misconception_confidence_is_bounded(confidence):
    with pytest.raises(ValidationError):
        Misconception(
            concept_id="c", description="d", confidence=confidence, first_seen_turn=0
        )


def test_hint_level_cannot_be_negative():
    with pytest.raises(ValidationError):
        LearnerState(session_id="s", hint_level=-1)


def test_empty_concept_id_is_rejected():
    with pytest.raises(ValidationError):
        ConceptMastery(concept_id="", score=0.5)


def test_retrieve_action_implies_retrieval():
    decision = PedagogicalDecision(
        action=PedagogicalAction.RETRIEVE, retrieval_query="interoperability"
    )
    assert decision.needs_retrieval is True


def test_retrieval_without_a_query_is_normalised_off():
    decision = PedagogicalDecision(
        action=PedagogicalAction.EXPLAIN_CONCEPT, needs_retrieval=True
    )
    assert decision.needs_retrieval is False


def test_diagnosis_lists_are_capped():
    from medical_ai_tutor.state.models import (
        Intent,
        LearnerDiagnosis,
        MasteryEvidence,
        ResponseQuality,
    )

    diagnosis = LearnerDiagnosis(
        intent=Intent.STUDENT_ATTEMPT,
        mastery_evidence=[
            MasteryEvidence(concept_id=f"c{i}", quality=ResponseQuality.CORRECT)
            for i in range(50)
        ],
    )
    assert len(diagnosis.mastery_evidence) == 8
