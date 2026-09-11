"""Learner state: persistence and bounded, deterministic updates."""

from __future__ import annotations

from medical_ai_tutor.state.models import (
    DetectedMisconception,
    Intent,
    LearnerDiagnosis,
    LearnerState,
    MasteryEvidence,
    Observation,
    PedagogicalAction,
    PedagogicalDecision,
    ResponseQuality,
)
from medical_ai_tutor.state.store import SessionStore
from medical_ai_tutor.state.updater import StateUpdater


def _diagnosis(quality: ResponseQuality, concept: str = "interoperability") -> LearnerDiagnosis:
    return LearnerDiagnosis(
        intent=Intent.STUDENT_ATTEMPT,
        response_quality=quality,
        mastery_evidence=[MasteryEvidence(concept_id=concept, quality=quality)],
        current_topic=concept,
    )


def test_state_persists_across_turns(orchestrator):
    orchestrator.run_turn("persist-1", "I think interoperability means exchanging messages.")
    first = orchestrator.get_state("persist-1")
    assert first is not None and first.turn_count == 1

    orchestrator.run_turn("persist-1", "I would add that the meaning has to survive the exchange.")
    second = orchestrator.get_state("persist-1")
    assert second.turn_count == 2
    assert len(second.dialogue) >= 4
    assert len(second.previous_actions) == 2


def test_state_survives_a_new_store_instance(config):
    db = str(config.path("storage.db_path"))
    store = SessionStore(db)
    state = store.create("reload-1")
    state.current_topic = "3lgm2"
    store.save(state)

    reopened = SessionStore(db).load("reload-1")
    assert reopened is not None
    assert reopened.current_topic == "3lgm2"


def test_mastery_stays_within_bounds_under_repeated_evidence(config):
    updater = StateUpdater(config)
    state = LearnerState(session_id="bounds")
    for turn in range(1, 40):
        state, _ = updater.apply(
            state, _diagnosis(ResponseQuality.CORRECT), Observation.from_student("x", turn)
        )
    assert 0.0 <= state.mastery_of("interoperability") <= 1.0

    for turn in range(40, 80):
        state, _ = updater.apply(
            state, _diagnosis(ResponseQuality.INCORRECT), Observation.from_student("x", turn)
        )
    assert 0.0 <= state.mastery_of("interoperability") <= 1.0


def test_updates_are_deterministic(config):
    updater = StateUpdater(config)
    results = []
    for _ in range(3):
        state = LearnerState(session_id="determinism")
        state, _ = updater.apply(
            state, _diagnosis(ResponseQuality.PARTIALLY_CORRECT), Observation.from_student("x", 1)
        )
        results.append(state.mastery_of("interoperability"))
    assert len(set(results)) == 1


def test_repeated_evidence_has_diminishing_impact(config):
    updater = StateUpdater(config)
    state = LearnerState(session_id="diminishing")
    deltas = []
    for turn in range(1, 5):
        before = state.mastery_of("interoperability")
        state, _ = updater.apply(
            state, _diagnosis(ResponseQuality.CORRECT), Observation.from_student("x", turn)
        )
        deltas.append(state.mastery_of("interoperability") - before)
    assert deltas == sorted(deltas, reverse=True)


def test_incorrect_evidence_does_not_zero_mastery(config):
    updater = StateUpdater(config)
    state = LearnerState(session_id="not-zero")
    for turn in range(1, 4):
        state, _ = updater.apply(
            state, _diagnosis(ResponseQuality.CORRECT), Observation.from_student("x", turn)
        )
    high = state.mastery_of("interoperability")
    state, _ = updater.apply(
        state, _diagnosis(ResponseQuality.INCORRECT), Observation.from_student("x", 4)
    )
    assert 0.0 < state.mastery_of("interoperability") < high


def test_misconception_lifecycle_keeps_history(config):
    updater = StateUpdater(config)
    state = LearnerState(session_id="misconception")

    detect = LearnerDiagnosis(
        intent=Intent.STUDENT_ATTEMPT,
        response_quality=ResponseQuality.PARTIALLY_CORRECT,
        detected_misconceptions=[
            DetectedMisconception(concept_id="normalization", description="arbitrary splitting")
        ],
    )
    state, delta = updater.apply(state, detect, Observation.from_student("x", 1))
    assert delta.misconceptions_added == ["normalization"]
    assert len(state.active_misconceptions()) == 1

    resolve = LearnerDiagnosis(
        intent=Intent.ANSWER_TO_TUTOR_QUESTION,
        response_quality=ResponseQuality.CORRECT,
        resolved_misconceptions=["normalization"],
    )
    state, delta = updater.apply(state, resolve, Observation.from_student("y", 2))
    state, delta = updater.apply(state, resolve, Observation.from_student("y", 3))
    assert state.find_misconception("normalization") is not None  # history retained
    assert state.active_misconceptions() == []


def test_hint_level_is_bounded_and_advances_one_step(config):
    updater = StateUpdater(config)
    state = LearnerState(session_id="hints")
    decision = PedagogicalDecision(action=PedagogicalAction.GIVE_HINT, desired_hint_level=99)

    for turn in range(1, 12):
        state = updater.record_action_and_response(state, decision, "a hint", turn)
    assert state.hint_level == updater.hint_max
    assert len(state.previous_hints) <= updater.max_previous_hints


def test_switching_exercise_resets_scaffolding(config):
    updater = StateUpdater(config)
    state = LearnerState(
        session_id="switch", current_exercise_id="1.4.1", hint_level=3, attempt_count=4
    )
    diagnosis = LearnerDiagnosis(intent=Intent.EXERCISE_HELP, current_exercise_id="1.4.2")
    state, _ = updater.apply(state, diagnosis, Observation.from_student("x", 1))
    assert state.current_exercise_id == "1.4.2"
    assert state.hint_level == 0
    assert state.attempt_count == 0


def test_dialogue_window_is_bounded_and_older_turns_are_summarised(config):
    updater = StateUpdater(config)
    state = LearnerState(session_id="window")
    for turn in range(1, 40):
        state, _ = updater.apply(
            state, _diagnosis(ResponseQuality.CORRECT), Observation.from_student(f"msg {turn}", turn)
        )
    assert len(state.dialogue) <= updater.max_dialogue
    assert state.history_summary
    assert len(state.history_summary) <= updater.summary_max_chars
