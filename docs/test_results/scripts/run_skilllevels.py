"""Skill-level replicability using the pytest live-test seeding (conf 0.8, evidence 4), 2 repeats."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]   # <repo>/docs/test_results/scripts -> <repo>
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)
from live_lib import make_tutor, session
from medical_ai_tutor.state.models import ConceptMastery

tutor = make_tutor("levels")
QUESTION = "What is interoperability in a health information system?"
out = {}
for rep in (1, 2):
    actions = {}
    print("\n" + "#" * 78); print(f"# REPEAT {rep}"); print("#" * 78)
    for level, score in [("beginner", 0.05), ("intermediate", 0.45), ("advanced", 0.9)]:
        s = session()
        st = tutor.create_session(s)
        st = st.model_copy(update={
            "current_topic": "interoperability",
            "concept_mastery": {"interoperability": ConceptMastery(
                concept_id="interoperability", score=score, confidence=0.8, evidence_count=4)},
            "demonstrated_concepts": ["interoperability"] if score > 0.7 else [],
        })
        tutor.sessions.save(st)
        resp, tr = tutor.run_turn(s, QUESTION)
        actions[level] = resp.action.value
        print(f"\n  {level} (mastery={score})")
        print(f"    action   : {resp.action.value}  ({tr.decision.reason_code})")
        print(f"    hint_lvl : desired={tr.decision.desired_hint_level}")
        print(f"    retrieval: {bool(resp.retrieved_section_ids)} {resp.retrieved_section_ids}")
        print(f"    citations: {resp.citations}")
        print(f"    safety   : {resp.safety_verdict.value}")
        print(f"    chars    : {len(resp.text)}")
        print(f"    reply    : {resp.text}")
    print(f"\n  >> actions repeat {rep}: {actions}  distinct={len(set(actions.values()))}")
    out[rep] = actions
print("\n\nBOTH REPEATS:"); print(json.dumps(out, indent=1))
(OUT_DIR / "skilllevels.json").write_text(json.dumps(out, indent=1))
