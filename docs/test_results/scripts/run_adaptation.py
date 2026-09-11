"""Replicability across skill levels: adaptation cases, live, with per-case detail."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]   # <repo>/docs/test_results/scripts -> <repo>
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)

from medical_ai_tutor.config import load_config
from medical_ai_tutor.evals.runner import EvalRunner, load_cases
from medical_ai_tutor.llm.factory import build_client

config = load_config()
client = build_client(config, "openai")
cases = [c for c in load_cases() if c.get("dimension") == "adaptation"]
print(f"cases: {[c['id'] for c in cases]}")

rows = []
for repeat in (1, 2):
    runner = EvalRunner(config=config, client=client, db_path=":memory:")
    summary = runner.run_all(cases)
    print("\n" + "#" * 78)
    print(f"# ADAPTATION PASS {repeat}")
    print("#" * 78)
    for r in runner.records:
        print("=" * 78)
        print(f"case      : {r.case_id}")
        print(f"message   : {r.message}")
        print(f"intent    : {r.intent}")
        print(f"action    : {r.action}  ({r.reason_code})")
        print(f"retrieval : used={r.retrieval_used} expected={r.retrieval_expected} sections={r.retrieved_sections}")
        print(f"safety    : {r.safety_verdict} | leaked={r.leaked}")
        print(f"checks    : {[(c['kind'], c['ok']) for c in r.checks]}")
        print(f"reply     : {r.response}")
        rows.append({"pass": repeat, "case_id": r.case_id, "action": r.action,
                     "reason_code": r.reason_code, "intent": r.intent,
                     "retrieval_used": r.retrieval_used,
                     "retrieval_expected": r.retrieval_expected,
                     "sections": r.retrieved_sections,
                     "safety": r.safety_verdict, "leaked": r.leaked,
                     "passed": r.passed, "checks": r.checks, "response": r.response})
    print(f"\nPASS {repeat} adaptation block: {json.dumps(summary['adaptation'], indent=1)}")
    print(f"PASS {repeat} routing: turns_passed={summary['routing']['turns_passed']}/{summary['routing']['turns']} "
          f"rate={summary['routing']['assertion_pass_rate']}")
    print(f"PASS {repeat} tools: {json.dumps(summary['tools'])}")
    print(f"PASS {repeat} failures: {json.dumps(summary['failures'], indent=1)}")

by_pass = {}
for r in rows:
    by_pass.setdefault(r["pass"], {})[r["case_id"]] = r["action"]
print("\n\nACTION BY LEVEL, BOTH PASSES:")
print(json.dumps(by_pass, indent=1))
(OUT_DIR / "adaptation.json").write_text(json.dumps(rows, indent=1))
