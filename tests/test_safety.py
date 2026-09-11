"""Protected-solution safety: isolation, detection, judging, revision, fallback."""

from __future__ import annotations

import json

import pytest

from medical_ai_tutor.llm.base import LLMResult
from medical_ai_tutor.llm.mock_client import MockLLMClient
from medical_ai_tutor.safety.deterministic import DeterministicDetector, normalize
from medical_ai_tutor.safety.gate import SafetyGate, most_severe
from medical_ai_tutor.safety.protected_store import (
    ProtectedAccessViolation,
    ProtectedStore,
)
from medical_ai_tutor.state.models import (
    LearnerState,
    PedagogicalAction,
    PedagogicalDecision,
    SafetyVerdict,
)
from medical_ai_tutor.tools.registry import FORBIDDEN_TOOL_NAMES, ToolPermissionError, ToolRegistry
from medical_ai_tutor.tutor.fallback import safe_fallback_text


@pytest.fixture
def store(ingested) -> ProtectedStore:
    return ProtectedStore.from_file(ingested["protected"] / "solutions.json")


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #


def test_protected_store_rejects_non_safety_accessors(store):
    for accessor in ("tutor_generator", "policy", "retrieval", "api", "student"):
        with pytest.raises(ProtectedAccessViolation):
            store.get("1.4.1", accessor=accessor)
    assert len(store.violations()) == 5


def test_safety_components_may_read_solutions(store):
    assert store.get("1.4.1", accessor="safety_gate") is not None
    assert store.get("1.4.1", accessor="semantic_judge") is not None
    assert store.violations() == []


def test_protected_tool_names_cannot_be_registered():
    registry = ToolRegistry()
    for name in FORBIDDEN_TOOL_NAMES:
        tool = type("T", (), {"name": name, "permissions": frozenset({"tutor"})})()
        with pytest.raises(ToolPermissionError):
            registry.register(tool)


def test_normal_tools_cannot_reach_protected_solutions(orchestrator, ingested):
    """The registry exposes no route to a solution, by name or by payload."""
    assert "get_protected_solution" not in orchestrator.tools
    assert set(orchestrator.tools.names()) == {"get_exercise", "search_course_material"}

    with open(ingested["protected"] / "solutions.json", encoding="utf-8") as handle:
        solutions = json.load(handle)

    orchestrator.tools.begin_turn()
    result = orchestrator.tools.execute("get_exercise", {"exercise_id": "1.4.1"})
    payload = json.dumps(result.payload)
    for solution in solutions:
        for unit in solution["answer_units"]:
            assert unit not in payload
    assert "solution" not in {k.lower() for k in result.payload}


def test_exercise_catalog_rejects_a_file_carrying_solutions(tmp_path):
    from medical_ai_tutor.tools.base import ToolError
    from medical_ai_tutor.tools.exercise_tool import ExerciseCatalog

    path = tmp_path / "exercises.json"
    path.write_text(
        json.dumps([{"exercise_id": "1.1.1", "question": "q", "solution_text": "leak"}]),
        encoding="utf-8",
    )
    with pytest.raises(ToolError):
        ExerciseCatalog.from_file(path)


# --------------------------------------------------------------------------- #
# Layer A: deterministic detection
# --------------------------------------------------------------------------- #


def test_detector_catches_a_verbatim_copy(store):
    solution = store.get("1.4.1", accessor="test")
    finding = DeterministicDetector().check(
        solution.solution_text, solution.solution_text, solution.answer_units, solution.unit_labels()
    )
    assert finding.leaked
    assert finding.covered_units
    assert finding.max_verbatim_run > 10
    assert "answer_units_covered" in finding.reason_codes


def test_detector_survives_cosmetic_obfuscation(store):
    solution = store.get("1.4.1", accessor="test")
    disguised = (
        solution.solution_text.replace(" ", "  ")
        .replace(".", " . ")
        .replace('"', "“")
        .upper()
    )
    assert DeterministicDetector().check(
        disguised, solution.solution_text, solution.answer_units, solution.unit_labels()
    ).leaked


def test_detector_passes_a_socratic_question(store):
    solution = store.get("1.4.1", accessor="test")
    finding = DeterministicDetector().check(
        "What would the physician need to already know for that reading to mean anything?",
        solution.solution_text,
        solution.answer_units,
        solution.unit_labels(),
    )
    assert not finding.leaked
    assert finding.covered_units == []


def test_normalisation_folds_case_accents_and_punctuation():
    assert normalize("Héllo, WORLD!") == normalize("hello world")
    assert normalize("“quoted”") == normalize("quoted")


def test_detector_returns_structured_evidence_not_a_boolean(store):
    solution = store.get("1.4.1", accessor="test")
    finding = DeterministicDetector().check(
        solution.solution_text, solution.solution_text, solution.answer_units, solution.unit_labels()
    )
    assert set(finding.model_dump()) >= {
        "leaked",
        "ngram_overlap",
        "max_verbatim_run",
        "covered_units",
        "unit_coverage",
        "reason_codes",
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def test_gate_is_inert_without_a_protected_exercise(config, store):
    gate = SafetyGate(config, store, MockLLMClient())
    solution = store.get("1.4.1", accessor="test")
    report = gate.evaluate(solution.solution_text, LearnerState(session_id="s"))
    assert report.verdict == SafetyVerdict.PASS
    assert report.reason_code == "no_protected_exercise"


def test_gate_blocks_a_full_disclosure(config, store):
    gate = SafetyGate(config, store, MockLLMClient())
    state = LearnerState(session_id="s", current_exercise_id="1.4.1")
    solution = store.get("1.4.1", accessor="test")
    assert gate.evaluate(solution.solution_text, state).verdict == SafetyVerdict.BLOCK


def test_deterministic_layer_cannot_be_overridden_by_a_lenient_judge(config, store):
    """A judge that always says PASS must not soften a measured leak."""

    class LenientClient(MockLLMClient):
        def _judge(self, context):  # type: ignore[override]
            from medical_ai_tutor.state.models import JudgeVerdict

            return JudgeVerdict(verdict=SafetyVerdict.PASS, reason_code="no_leakage")

    gate = SafetyGate(config, store, LenientClient())
    state = LearnerState(session_id="s", current_exercise_id="1.4.1")
    solution = store.get("1.4.1", accessor="test")
    report = gate.evaluate(solution.solution_text, state)
    assert report.verdict == SafetyVerdict.BLOCK
    assert report.judge is not None and report.judge.verdict == SafetyVerdict.PASS


def test_judge_failure_does_not_become_an_implicit_pass(config, store):
    class BrokenJudge(MockLLMClient):
        def structured(self, system, messages, schema, model, **kwargs):  # type: ignore[override]
            if "ROLE: safety_judge" in system:
                raise RuntimeError("provider down")
            return super().structured(system, messages, schema, model, **kwargs)

    gate = SafetyGate(config, store, BrokenJudge())
    state = LearnerState(session_id="s", current_exercise_id="1.4.1")
    report = gate.evaluate("A perfectly innocuous sentence.", state)
    assert report.judge is not None
    assert report.judge.reason_code == "judge_unavailable"
    assert report.verdict == SafetyVerdict.REVISE


def test_verdict_severity_ordering():
    assert most_severe(SafetyVerdict.PASS, SafetyVerdict.REVISE) == SafetyVerdict.REVISE
    assert most_severe(SafetyVerdict.REVISE, SafetyVerdict.BLOCK) == SafetyVerdict.BLOCK
    assert most_severe(SafetyVerdict.PASS, SafetyVerdict.PASS) == SafetyVerdict.PASS


def test_gate_never_puts_solution_text_in_its_report(config, store):
    gate = SafetyGate(config, store, MockLLMClient())
    state = LearnerState(session_id="s", current_exercise_id="1.4.1")
    solution = store.get("1.4.1", accessor="test")
    report = gate.evaluate(solution.solution_text, state)

    labels = json.dumps(report.leaked_units)
    for unit in solution.answer_units:
        assert unit not in labels


# --------------------------------------------------------------------------- #
# Revision and fallback
# --------------------------------------------------------------------------- #


class LeakyClient(MockLLMClient):
    """A generator that always emits the protected solution, however often asked."""

    def __init__(self, solution_text: str) -> None:
        super().__init__()
        self.solution_text = solution_text
        self.generation_calls = 0
        self.revision_calls = 0

    def complete(self, system, messages, model, temperature=0.2, max_tokens=1024) -> LLMResult:
        role = self._role_of(system)
        if role == "revision":
            self.revision_calls += 1
        else:
            self.generation_calls += 1
        return LLMResult(text=self.solution_text, model=model)


def test_revision_loop_stops_at_the_configured_limit(config, store, ingested):
    from medical_ai_tutor.tutor.revision import RevisionLoop

    solution = store.get("1.4.1", accessor="test")
    client = LeakyClient(solution.solution_text)
    gate = SafetyGate(config, store, client)
    loop = RevisionLoop(client, config, gate)

    state = LearnerState(session_id="s", current_exercise_id="1.4.1")
    decision = PedagogicalDecision(action=PedagogicalAction.GIVE_HINT, target_concept="x")
    report = gate.evaluate(solution.solution_text, state)

    text, final_report, records, used_fallback, _, _ = loop.run(
        candidate=solution.solution_text,
        report=report,
        state=state,
        diagnosis=_blank_diagnosis(),
        decision=decision,
        passages=[],
        exercise=None,
    )
    max_revisions = int(config.get("limits.max_revisions_per_turn"))
    assert len(records) <= max_revisions
    assert client.revision_calls <= max_revisions
    assert used_fallback is True
    assert final_report.verdict == SafetyVerdict.PASS
    assert text == safe_fallback_text(decision, state)


def test_safe_fallback_is_emitted_and_is_itself_clean(config, store):
    solution = store.get("1.4.1", accessor="test")
    gate = SafetyGate(config, store, MockLLMClient())
    state = LearnerState(session_id="s", current_exercise_id="1.4.1")

    for action in PedagogicalAction:
        decision = PedagogicalDecision(action=action, target_concept="interoperability")
        text = safe_fallback_text(decision, state)
        assert text and len(text.split()) > 5
        assert gate.evaluate(text, state).verdict == SafetyVerdict.PASS
        for unit in solution.answer_units:
            assert unit not in text


def test_a_leaky_generator_never_reaches_the_learner(config, store, ingested):
    """End-to-end: even a generator that only emits the solution is contained."""
    from medical_ai_tutor.tutor.orchestrator import build_orchestrator

    solution = store.get("1.4.1", accessor="test")
    client = LeakyClient(solution.solution_text)
    tutor = build_orchestrator(config, client=client)

    response, trace = tutor.run_turn("leaky", "Help me with this exercise.", exercise_id="1.4.1")

    detector = DeterministicDetector()
    finding = detector.check(
        response.text, solution.solution_text, solution.answer_units, solution.unit_labels()
    )
    assert not finding.leaked
    assert response.used_fallback is True
    assert trace.safety_verdict == SafetyVerdict.PASS


def _blank_diagnosis():
    from medical_ai_tutor.state.models import Intent, LearnerDiagnosis

    return LearnerDiagnosis(intent=Intent.EXERCISE_HELP)
