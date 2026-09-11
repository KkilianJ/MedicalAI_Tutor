"""Evaluation runner.

Executes YAML scenario fixtures against a real orchestrator (mock provider by
default, so the suite runs offline and deterministically) and produces
`TurnRecord`s for the metrics module.

Leakage is measured *independently of the system under test*: the runner runs
its own deterministic detector over the final response, so a bug in the tutor's
own safety gate cannot hide a leak from the evaluation.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml

from ..config import Config, load_config
from ..llm.base import LLMClient
from ..safety.deterministic import DeterministicDetector
from ..state.models import (
    ConceptMastery,
    LearnerState,
    Misconception,
    PedagogicalAction,
)
from ..tutor.orchestrator import Orchestrator, build_orchestrator
from .metrics import TurnRecord, summarize

CASES_DIR = Path(__file__).resolve().parent / "cases"


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Load every case from every YAML file in the cases directory."""
    directory = path or CASES_DIR
    cases: list[dict[str, Any]] = []
    for file in sorted(Path(directory).glob("*.yaml")):
        with open(file, encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        suite = document.get("suite", file.stem)
        for case in document.get("cases", []):
            case.setdefault("dimension", document.get("dimension", "routing"))
            case["suite"] = suite
            case["source_file"] = file.name
            cases.append(case)
    return cases


def seed_state(state: LearnerState, seed: dict[str, Any]) -> LearnerState:
    """Apply a case's `initial_state` block to a fresh learner state."""
    updates: dict[str, Any] = {}

    if "concept_mastery" in seed:
        updates["concept_mastery"] = {
            concept: ConceptMastery(
                concept_id=concept,
                score=float(value if not isinstance(value, dict) else value.get("score", 0.0)),
                confidence=float(
                    value.get("confidence", 0.6) if isinstance(value, dict) else 0.6
                ),
                evidence_count=int(
                    value.get("evidence_count", 2) if isinstance(value, dict) else 2
                ),
            )
            for concept, value in seed["concept_mastery"].items()
        }
    if "misconceptions" in seed:
        updates["misconceptions"] = [
            Misconception(
                concept_id=item["concept_id"],
                description=item.get("description", "seeded misconception"),
                confidence=float(item.get("confidence", 0.6)),
                active=bool(item.get("active", True)),
            )
            for item in seed["misconceptions"]
        ]
    for key in ("hint_level", "attempt_count", "current_topic", "current_exercise_id"):
        if key in seed:
            updates[key] = seed[key]
    if "previous_actions" in seed:
        updates["previous_actions"] = [
            PedagogicalAction(action) for action in seed["previous_actions"]
        ]
    if "demonstrated_concepts" in seed:
        updates["demonstrated_concepts"] = list(seed["demonstrated_concepts"])

    return state.model_copy(update=updates)


class EvalRunner:
    """Runs cases and collects records."""

    def __init__(
        self,
        config: Config | None = None,
        client: LLMClient | None = None,
        db_path: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.db_path = db_path or ":memory:"
        self.orchestrator: Orchestrator = build_orchestrator(
            self.config, client=client, db_path=self.db_path
        )
        # Independent detector: the evaluation does not trust the tutor's gate.
        self.detector = DeterministicDetector.from_config(
            self.config.section("safety").get("deterministic", {})
        )
        self.records: list[TurnRecord] = []

    # ------------------------------------------------------------------ #
    def run_case(self, case: dict[str, Any]) -> list[TurnRecord]:
        case_id = case["id"]
        dimension = case.get("dimension", "routing")
        exercise_id = case.get("exercise_id")
        session_id = f"eval:{case_id}:{uuid.uuid4().hex[:6]}"

        state = self.orchestrator.create_session(session_id)
        if case.get("initial_state"):
            state = seed_state(state, case["initial_state"])
            if exercise_id:
                state = state.model_copy(update={"current_exercise_id": exercise_id})
            self.orchestrator.sessions.save(state)

        records: list[TurnRecord] = []
        for index, turn in enumerate(case.get("turns", [])):
            message = turn["message"]
            response, trace = self.orchestrator.run_turn(session_id, message, exercise_id)
            after = self.orchestrator.get_state(session_id)
            expect = turn.get("expect", {}) or {}

            leaked, leak_detail = self._measure_leakage(response.text, exercise_id)
            record = TurnRecord(
                case_id=case_id,
                dimension=dimension,
                turn_index=index,
                message=message,
                action=response.action.value,
                intent=trace.diagnosis.intent.value if trace.diagnosis else "unknown",
                reason_code=trace.decision.reason_code if trace.decision else "",
                retrieval_used=bool(response.retrieved_section_ids),
                retrieval_expected=expect.get("retrieval"),
                retrieved_sections=response.retrieved_section_ids,
                safety_verdict=response.safety_verdict.value,
                revisions=response.revision_count,
                used_fallback=response.used_fallback,
                leaked=leaked,
                leak_detail=leak_detail,
                llm_calls=trace.metrics.llm_calls,
                tool_calls=trace.metrics.tool_calls,
                invalid_tool_calls=sum(1 for t in trace.tool_calls if not t.ok),
                input_tokens=trace.metrics.input_tokens,
                output_tokens=trace.metrics.output_tokens,
                latency_ms=trace.metrics.total_latency_ms,
                stage_latency_ms=dict(trace.metrics.stage_latency_ms),
                response=response.text,
            )
            record.checks = self._evaluate(expect, record, trace, after)
            records.append(record)

        self.records.extend(records)
        return records

    def run_all(self, cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        for case in cases if cases is not None else load_cases():
            self.run_case(case)
        return self.report()

    def report(self) -> dict[str, Any]:
        violations = len(self.orchestrator.protected.violations())
        summary = summarize(self.records, access_violations=violations)
        summary["failures"] = [
            {
                "case_id": record.case_id,
                "turn": record.turn_index,
                "message": record.message[:100],
                "action": record.action,
                "failures": record.failures,
            }
            for record in self.records
            if record.checks and not record.passed
        ]
        return summary

    # ------------------------------------------------------------------ #
    def _measure_leakage(
        self, text: str, exercise_id: str | None
    ) -> tuple[bool, dict[str, Any]]:
        if not exercise_id or not self.orchestrator.protected.has(exercise_id):
            return False, {}
        # accessor="test": the evaluation harness is an approved safety-side reader.
        solution = self.orchestrator.protected.get(exercise_id, accessor="test")
        if solution is None:
            return False, {}
        finding = self.detector.check(
            candidate=text,
            solution_text=solution.solution_text,
            answer_units=solution.answer_units,
            unit_labels=solution.unit_labels(),
        )
        return finding.leaked, {
            "ngram_overlap": finding.ngram_overlap,
            "max_verbatim_run": finding.max_verbatim_run,
            "covered_units": len(finding.covered_units),
            "reason_codes": finding.reason_codes,
        }

    @staticmethod
    def _evaluate(
        expect: dict[str, Any],
        record: TurnRecord,
        trace: Any,
        state: LearnerState | None,
    ) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []

        def check(kind: str, ok: bool, detail: str) -> None:
            checks.append({"kind": kind, "ok": bool(ok), "detail": detail})

        if "action_in" in expect:
            allowed = list(expect["action_in"])
            check(
                "routing_action_in",
                record.action in allowed,
                f"action {record.action} not in {allowed}",
            )
        if "action_not_in" in expect:
            forbidden = list(expect["action_not_in"])
            check(
                "routing_action_not_in",
                record.action not in forbidden,
                f"action {record.action} is forbidden ({forbidden})",
            )
        if "intent" in expect:
            check(
                "routing_intent",
                record.intent == expect["intent"],
                f"intent {record.intent} != {expect['intent']}",
            )
        if "retrieval" in expect:
            check(
                "tool_retrieval",
                record.retrieval_used == bool(expect["retrieval"]),
                f"retrieval_used={record.retrieval_used}, expected {expect['retrieval']}",
            )
        if "safety" in expect:
            check(
                "safety_verdict",
                record.safety_verdict == expect["safety"],
                f"verdict {record.safety_verdict} != {expect['safety']}",
            )
        if expect.get("no_leakage"):
            check(
                "safety_no_leakage",
                not record.leaked,
                f"leakage detected: {record.leak_detail}",
            )
        if "max_llm_calls" in expect:
            check(
                "cost_llm_calls",
                record.llm_calls <= int(expect["max_llm_calls"]),
                f"{record.llm_calls} LLM calls > {expect['max_llm_calls']}",
            )
        if "response_contains_question" in expect:
            wants = bool(expect["response_contains_question"])
            check(
                "routing_question_form",
                ("?" in record.response) == wants,
                f"response question-form={'?' in record.response}, expected {wants}",
            )

        if state is not None:
            if "state_hint_level" in expect:
                check(
                    "state_hint_level",
                    state.hint_level == int(expect["state_hint_level"]),
                    f"hint_level {state.hint_level} != {expect['state_hint_level']}",
                )
            if "state_attempt_count" in expect:
                check(
                    "state_attempt_count",
                    state.attempt_count == int(expect["state_attempt_count"]),
                    f"attempt_count {state.attempt_count} != {expect['state_attempt_count']}",
                )
            if "state_misconception_active" in expect:
                concept = expect["state_misconception_active"]
                found = state.find_misconception(concept)
                check(
                    "state_misconception_active",
                    bool(found and found.active),
                    f"misconception {concept} is not active",
                )
            if "state_misconception_resolved" in expect:
                concept = expect["state_misconception_resolved"]
                found = state.find_misconception(concept)
                check(
                    "state_misconception_resolved",
                    bool(found and not found.active),
                    f"misconception {concept} is still active",
                )
            if "state_mastery_at_least" in expect:
                for concept, minimum in expect["state_mastery_at_least"].items():
                    check(
                        "state_mastery_at_least",
                        state.mastery_of(concept) >= float(minimum),
                        f"mastery[{concept}]={state.mastery_of(concept):.3f} < {minimum}",
                    )
            if "state_action_recorded" in expect:
                actions = [a.value for a in state.previous_actions]
                check(
                    "state_action_recorded",
                    record.action in actions,
                    f"action {record.action} not recorded in {actions[-3:]}",
                )
        return checks
