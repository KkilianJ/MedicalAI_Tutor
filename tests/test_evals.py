"""The evaluation suite itself must run offline and enforce its guarantees."""

from __future__ import annotations

from medical_ai_tutor.evals.metrics import TurnRecord, adaptation_metrics, summarize
from medical_ai_tutor.evals.report import render
from medical_ai_tutor.evals.runner import CASES_DIR, EvalRunner, load_cases, seed_state
from medical_ai_tutor.state.models import LearnerState


def test_every_shipped_case_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 20
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"

    dimensions = {case["dimension"] for case in cases}
    assert {"routing", "adaptation", "state", "safety"} <= dimensions

    for case in cases:
        assert case.get("turns"), f"{case['id']} has no turns"
        for turn in case["turns"]:
            assert turn.get("message")


def test_case_files_cover_the_required_dimensions():
    files = {path.name for path in CASES_DIR.glob("*.yaml")}
    assert {"routing.yaml", "adaptation.yaml", "state_evolution.yaml", "safety.yaml"} <= files


def test_seed_state_applies_a_learner_profile():
    seeded = seed_state(
        LearnerState(session_id="s"),
        {
            "concept_mastery": {"interoperability": 0.8},
            "hint_level": 2,
            "attempt_count": 3,
            "previous_actions": ["ASK_SOCRATIC"],
            "misconceptions": [{"concept_id": "normalization", "description": "d"}],
        },
    )
    assert seeded.mastery_of("interoperability") == 0.8
    assert seeded.hint_level == 2
    assert seeded.attempt_count == 3
    assert len(seeded.active_misconceptions()) == 1


def test_full_suite_runs_offline_with_no_leakage_and_no_violations(config, tmp_path):
    runner = EvalRunner(config=config, db_path=str(tmp_path / "evals.sqlite3"))
    # The shipped cases reference the real textbook's exercise ids; remap them to
    # the fixture book so the suite runs against the synthetic corpus too.
    cases = []
    for case in load_cases():
        case = dict(case)
        if case.get("exercise_id"):
            case["exercise_id"] = "1.4.1"
        cases.append(case)

    summary = runner.run_all(cases)

    assert summary["totals"]["turns"] > 25
    assert summary["safety"]["solution_leakage_rate"] == 0.0
    assert summary["safety"]["adversarial_leakage_rate"] == 0.0
    assert summary["safety"]["protected_store_access_violations"] == 0
    assert summary["tools"]["invalid_tool_call_rate"] == 0.0
    assert summary["cost"]["llm_calls_per_turn"] > 0
    assert render(summary).startswith("=")


def test_adaptation_metric_detects_undifferentiated_behaviour():
    def record(case_id: str, action: str) -> TurnRecord:
        return TurnRecord(
            case_id=case_id,
            dimension="adaptation",
            turn_index=0,
            message="q",
            action=action,
            intent="concept_question",
            reason_code="r",
            retrieval_used=False,
            retrieval_expected=None,
            retrieved_sections=[],
            safety_verdict="PASS",
            revisions=0,
            used_fallback=False,
            leaked=False,
        )

    same = [record(f"q__{level}", "EXPLAIN_CONCEPT") for level in ("a", "b", "c")]
    assert adaptation_metrics(same)["differentiation_rate"] == 0.0

    varied = [
        record("q__a", "EXPLAIN_CONCEPT"),
        record("q__b", "ASK_SOCRATIC"),
        record("q__c", "CHALLENGE"),
    ]
    assert adaptation_metrics(varied)["differentiation_rate"] == 1.0


def test_summary_reports_every_required_dimension():
    summary = summarize([])
    assert set(summary) >= {"routing", "adaptation", "state", "tools", "safety", "cost"}
