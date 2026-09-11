#!/usr/bin/env python3
"""Run the offline evaluation suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_ai_tutor.config import load_config  # noqa: E402
from medical_ai_tutor.evals.report import render  # noqa: E402
from medical_ai_tutor.evals.runner import EvalRunner, load_cases  # noqa: E402
from medical_ai_tutor.llm.factory import build_client_or_mock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, help="directory of case YAML files")
    parser.add_argument("--dimension", help="run only one dimension (routing/adaptation/state/safety)")
    parser.add_argument("--provider", help="override the configured provider")
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    parser.add_argument("--db", default=":memory:", help="SQLite path for eval sessions")
    parser.add_argument(
        "--fail-on-leak",
        action="store_true",
        help="exit non-zero if any leakage or access violation is measured",
    )
    args = parser.parse_args()

    config = load_config()
    # The suite is a regression gate, so it defaults to the deterministic mock:
    # it must be free, offline and reproducible. Hitting a real provider is an
    # explicit opt-in, because 37 turns is ~150 billed calls.
    provider = args.provider or "mock"
    client, warning = build_client_or_mock(config, provider)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    if provider != "mock":
        print(f"running against provider {provider!r} — this makes real API calls\n", file=sys.stderr)

    cases = load_cases(args.cases)
    if args.dimension:
        cases = [case for case in cases if case.get("dimension") == args.dimension]
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 1

    runner = EvalRunner(config=config, client=client, db_path=args.db)
    summary = runner.run_all(cases)
    print(render(summary))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nsummary written to {args.json}")

    safety = summary["safety"]
    leaked = safety["solution_leakage_rate"] > 0 or safety["protected_store_access_violations"] > 0
    if args.fail_on_leak and leaked:
        print("\nFAIL: leakage or protected-store violation detected", file=sys.stderr)
        return 2
    return 1 if summary["totals"]["failed_turns"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
