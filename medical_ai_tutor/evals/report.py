"""Human-readable evaluation report."""

from __future__ import annotations

import json
from typing import Any


def _line(label: str, value: Any, width: int = 34) -> str:
    return f"  {label.ljust(width)} {value}"


def render(summary: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append

    totals = summary.get("totals", {})
    add("=" * 74)
    add("MEDICAL AI TUTOR — EVALUATION REPORT")
    add("=" * 74)
    add(_line("turns evaluated", totals.get("turns", 0)))
    add(_line("failed turns", totals.get("failed_turns", 0)))
    add("")

    routing = summary.get("routing", {})
    add("A. PEDAGOGICAL ROUTING")
    add(_line("assertion pass rate", routing.get("assertion_pass_rate")))
    add(_line("turns passed", f"{routing.get('turns_passed')}/{routing.get('turns_with_assertions')}"))
    add(_line("action distribution", json.dumps(routing.get("action_distribution", {}))))
    add("")

    adaptation = summary.get("adaptation", {})
    add("B. LEARNER ADAPTATION")
    add(_line("adaptation groups", adaptation.get("adaptation_groups")))
    add(_line("differentiated groups", adaptation.get("differentiated_groups")))
    add(_line("differentiation rate", adaptation.get("differentiation_rate")))
    for question, detail in (adaptation.get("detail") or {}).items():
        add(f"    {question}: {json.dumps(detail['actions'])}")
    add("")

    state = summary.get("state", {})
    add("C. STATE EVOLUTION")
    add(_line("state assertions", state.get("state_assertions")))
    add(_line("state assertion pass rate", state.get("state_assertion_pass_rate")))
    add("")

    tools = summary.get("tools", {})
    add("D. TOOL BEHAVIOUR")
    add(_line("retrieval call rate", tools.get("retrieval_call_rate")))
    add(_line("unnecessary retrieval rate", tools.get("unnecessary_retrieval_rate")))
    add(_line("missed retrieval rate", tools.get("missed_retrieval_rate")))
    add(_line("invalid tool call rate", tools.get("invalid_tool_call_rate")))
    add("")

    safety = summary.get("safety", {})
    add("E. SAFETY")
    add(_line("solution leakage rate", safety.get("solution_leakage_rate")))
    add(_line("adversarial turns", safety.get("adversarial_turns")))
    add(_line("adversarial leakage rate", safety.get("adversarial_leakage_rate")))
    add(_line("revision rate", safety.get("revision_rate")))
    add(_line("fallback rate", safety.get("fallback_rate")))
    add(_line("protected access violations", safety.get("protected_store_access_violations")))
    add(_line("verdict distribution", json.dumps(safety.get("verdict_distribution", {}))))
    add("")

    cost = summary.get("cost", {})
    add("F. COST / EFFICIENCY")
    add(_line("LLM calls per turn", cost.get("llm_calls_per_turn")))
    add(_line("tool calls per turn", cost.get("tool_calls_per_turn")))
    add(_line("tokens (in -> out)", f"{cost.get('input_tokens_total')} -> {cost.get('output_tokens_total')}"))
    add(_line("latency mean / p95 (ms)", f"{cost.get('latency_ms_mean')} / {cost.get('latency_ms_p95')}"))
    add(_line("stage latency (ms)", json.dumps(cost.get("stage_latency_ms_mean", {}))))
    add("")

    failures = summary.get("failures", [])
    if failures:
        add(f"FAILURES ({len(failures)})")
        for failure in failures:
            add(f"  - {failure['case_id']} turn {failure['turn']} [{failure['action']}]")
            for detail in failure["failures"]:
                add(f"      {detail}")
    else:
        add("FAILURES: none")
    add("=" * 74)
    return "\n".join(out)
