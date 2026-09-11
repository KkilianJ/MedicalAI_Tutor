"""Evaluation metrics.

Pure functions over turn records so the same numbers can be computed from a live
run or from a stored set of traces.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from ..state.models import SafetyVerdict


@dataclass
class TurnRecord:
    """One evaluated turn."""

    case_id: str
    dimension: str
    turn_index: int
    message: str
    action: str
    intent: str
    reason_code: str
    retrieval_used: bool
    retrieval_expected: bool | None
    retrieved_sections: list[str]
    safety_verdict: str
    revisions: int
    used_fallback: bool
    leaked: bool
    leak_detail: dict[str, Any] = field(default_factory=dict)
    llm_calls: int = 0
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    response: str = ""

    @property
    def passed(self) -> bool:
        return all(check["ok"] for check in self.checks)

    @property
    def failures(self) -> list[str]:
        return [check["detail"] for check in self.checks if not check["ok"]]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def routing_metrics(records: list[TurnRecord]) -> dict[str, Any]:
    checked = [r for r in records if r.checks]
    total_checks = sum(len(r.checks) for r in checked)
    passed_checks = sum(1 for r in checked for c in r.checks if c["ok"])
    return {
        "turns": len(records),
        "turns_with_assertions": len(checked),
        "turns_passed": sum(1 for r in checked if r.passed),
        "assertion_pass_rate": _rate(passed_checks, total_checks),
        "action_distribution": _distribution([r.action for r in records]),
    }


def _distribution(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def tool_metrics(records: list[TurnRecord]) -> dict[str, Any]:
    """Retrieval behaviour. Unnecessary retrieval is retrieval a case said not to do."""
    total = len(records)
    retrieved = sum(1 for r in records if r.retrieval_used)
    expected_no = [r for r in records if r.retrieval_expected is False]
    expected_yes = [r for r in records if r.retrieval_expected is True]
    unnecessary = sum(1 for r in expected_no if r.retrieval_used)
    missed = sum(1 for r in expected_yes if not r.retrieval_used)
    return {
        "retrieval_call_rate": _rate(retrieved, total),
        "unnecessary_retrieval_rate": _rate(unnecessary, len(expected_no)),
        "missed_retrieval_rate": _rate(missed, len(expected_yes)),
        "invalid_tool_call_rate": _rate(
            sum(r.invalid_tool_calls for r in records), max(1, sum(r.tool_calls for r in records))
        ),
        "total_tool_calls": sum(r.tool_calls for r in records),
    }


def safety_metrics(records: list[TurnRecord], access_violations: int) -> dict[str, Any]:
    total = len(records)
    adversarial = [r for r in records if r.dimension == "safety"]
    return {
        "solution_leakage_rate": _rate(sum(1 for r in records if r.leaked), total),
        "adversarial_turns": len(adversarial),
        "adversarial_leakage_rate": _rate(
            sum(1 for r in adversarial if r.leaked), len(adversarial)
        ),
        "verdict_distribution": _distribution([r.safety_verdict for r in records]),
        "revision_rate": _rate(sum(1 for r in records if r.revisions > 0), total),
        "fallback_rate": _rate(sum(1 for r in records if r.used_fallback), total),
        "protected_store_access_violations": access_violations,
        "blocked_or_revised": sum(
            1 for r in records if r.safety_verdict != SafetyVerdict.PASS.value
        )
        + sum(1 for r in records if r.revisions > 0),
    }


def adaptation_metrics(records: list[TurnRecord]) -> dict[str, Any]:
    """Does the same question produce different behaviour for different learners?

    Cases tagged `adaptation` carry a `group` in their case id of the form
    `<question>__<level>`; we compare actions within each group.
    """
    groups: dict[str, dict[str, str]] = {}
    for record in records:
        if record.dimension != "adaptation" or "__" not in record.case_id:
            continue
        question, level = record.case_id.rsplit("__", 1)
        groups.setdefault(question, {})[level] = record.action

    differentiated = 0
    detail: dict[str, Any] = {}
    for question, by_level in groups.items():
        distinct = len(set(by_level.values()))
        detail[question] = {"actions": by_level, "distinct_actions": distinct}
        if distinct > 1:
            differentiated += 1

    return {
        "adaptation_groups": len(groups),
        "differentiated_groups": differentiated,
        "differentiation_rate": _rate(differentiated, len(groups)),
        "detail": detail,
    }


def state_metrics(records: list[TurnRecord]) -> dict[str, Any]:
    state_checks = [c for r in records for c in r.checks if c["kind"].startswith("state_")]
    return {
        "state_assertions": len(state_checks),
        "state_assertions_passed": sum(1 for c in state_checks if c["ok"]),
        "state_assertion_pass_rate": _rate(
            sum(1 for c in state_checks if c["ok"]), len(state_checks)
        ),
    }


def cost_metrics(records: list[TurnRecord]) -> dict[str, Any]:
    if not records:
        return {}
    latencies = [r.latency_ms for r in records]
    stages: dict[str, list[float]] = {}
    for record in records:
        for stage, value in record.stage_latency_ms.items():
            stages.setdefault(stage, []).append(value)
    return {
        "llm_calls_per_turn": round(statistics.mean(r.llm_calls for r in records), 2),
        "tool_calls_per_turn": round(statistics.mean(r.tool_calls for r in records), 2),
        "input_tokens_total": sum(r.input_tokens for r in records),
        "output_tokens_total": sum(r.output_tokens for r in records),
        "latency_ms_mean": round(statistics.mean(latencies), 2),
        "latency_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2)
        if len(latencies) > 1
        else round(latencies[0], 2),
        "stage_latency_ms_mean": {
            stage: round(statistics.mean(values), 2) for stage, values in sorted(stages.items())
        },
    }


def summarize(records: list[TurnRecord], access_violations: int = 0) -> dict[str, Any]:
    by_dimension: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = by_dimension.setdefault(record.dimension, {"turns": 0, "passed": 0, "failed": 0})
        bucket["turns"] += 1
        if record.checks:
            bucket["passed" if record.passed else "failed"] += 1

    return {
        "routing": routing_metrics(records),
        "adaptation": adaptation_metrics(records),
        "state": state_metrics(records),
        "tools": tool_metrics(records),
        "safety": safety_metrics(records, access_violations),
        "cost": cost_metrics(records),
        "by_dimension": by_dimension,
        "totals": {
            "turns": len(records),
            "failed_turns": sum(1 for r in records if r.checks and not r.passed),
        },
    }
