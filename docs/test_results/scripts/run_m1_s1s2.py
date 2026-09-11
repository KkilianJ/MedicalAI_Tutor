"""M1 (respond to the actual message) and S1/S2 (context + citations), live."""
from live_lib import make_tutor, session, show

tutor = make_tutor("m1s1s2")

print("\n" + "=" * 78)
print("M1  greeting -> should acknowledge and invite a course question")
print("=" * 78)
s = session()
msg = "Hi!"
r, t = tutor.run_turn(s, msg)
show("M1 turn 1 (greeting)", msg, r, t)

msg2 = "What topics can you help me with?"
r2, t2 = tutor.run_turn(s, msg2)
show("M1 turn 2 (follow-up)", msg2, r2, t2)

print("\n" + "=" * 78)
print("S1 + S2  misconception across turns, with course citations")
print("=" * 78)
s2 = session()
turns = [
    "I think an EHR and an EPR are just two names for exactly the same thing.",
    "So if they are different, does that mean the EPR is the one that belongs to a single hospital?",
    "Can you point me to where the book explains the difference?",
]
state_before = None
for i, m in enumerate(turns, 1):
    resp, tr = tutor.run_turn(s2, m)
    show(f"S1/S2 turn {i}", m, resp, tr)
    print(f"  delta     : {tr.state_delta.mastery_changes} "
          f"+mis={tr.state_delta.misconceptions_added} -mis={tr.state_delta.misconceptions_resolved}")

st = tutor.get_state(s2)
print(f"\n  final turn_count      : {st.turn_count}")
print(f"  final mastery         : { {k: round(v.score,3) for k,v in st.concept_mastery.items()} }")
print(f"  final misconceptions  : {[(m.concept_id, round(m.confidence,2), m.active) for m in st.misconceptions]}")
print(f"  previous_actions      : {[a.value for a in st.previous_actions]}")
print(f"  current_topic         : {st.current_topic}")
