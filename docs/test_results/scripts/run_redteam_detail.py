"""Re-run the safety cases and dump per-case detail (run 2 of the red-team suite)."""
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
cases = [c for c in load_cases() if c.get("dimension") == "safety"]
runner = EvalRunner(config=config, client=client, db_path=":memory:")
summary = runner.run_all(cases)

rows = []
for r in runner.records:
    print("=" * 78)
    print(f"case      : {r.case_id}  (turn {r.turn_index})")
    print(f"message   : {r.message}")
    print(f"intent    : {r.intent}")
    print(f"action    : {r.action}  ({r.reason_code})")
    print(f"safety    : {r.safety_verdict} | revisions={r.revisions} | fallback={r.used_fallback}")
    print(f"leaked    : {r.leaked}  detail={r.leak_detail}")
    print(f"checks    : {[(c['kind'], c['ok']) for c in r.checks]}")
    print(f"reply     : {r.response}")
    rows.append({
        "case_id": r.case_id, "turn": r.turn_index, "message": r.message,
        "intent": r.intent, "action": r.action, "reason_code": r.reason_code,
        "safety": r.safety_verdict, "leaked": r.leaked, "leak_detail": r.leak_detail,
        "passed": r.passed, "checks": r.checks, "response": r.response,
    })

print("\n" + "=" * 78)
print("SUMMARY (run 2)")
print(json.dumps({
    "turns": summary["totals"],
    "safety": summary["safety"],
    "routing": {k: v for k, v in summary["routing"].items() if k != "action_distribution"},
    "action_distribution": summary["routing"]["action_distribution"],
    "failures": summary["failures"],
}, indent=1))
(OUT_DIR / "redteam_detail.json").write_text(json.dumps(rows, indent=1))
