"""What the learner has covered — the learner-facing view of the learner model."""

from __future__ import annotations

import streamlit as st

from medical_ai_tutor.app.ui import (
    action_label,
    get_tutor,
    learner_state,
    mastery_band,
    readable,
)

tutor = get_tutor()
state = learner_state()

st.title("Your progress")

if state is None or state.turn_count == 0:
    st.caption("Nothing yet — head to **Learn** and ask something.")
    st.stop()

stats = st.container(horizontal=True)
stats.metric("Exchanges", state.turn_count)
stats.metric("Concepts touched", len(state.concept_mastery))
stats.metric("Attempts made", state.attempt_count)

# --- concepts ---------------------------------------------------------------
if state.concept_mastery:
    st.subheader("Concepts")
    st.caption(
        "These are my best guess from our conversation, not a grade. They move as "
        "you explain things."
    )
    for concept_id, mastery in sorted(
        state.concept_mastery.items(), key=lambda kv: -kv[1].score
    ):
        label, colour = mastery_band(mastery.score)
        with st.container(border=True):
            row = st.container(horizontal=True, vertical_alignment="center")
            row.markdown(f"**{readable(concept_id)}**")
            row.badge(label, color=colour)
            st.progress(min(1.0, mastery.score))
            st.caption(
                f"{mastery.evidence_count} thing"
                f"{'s' if mastery.evidence_count != 1 else ''} you said pointed at this"
            )

# --- what's being worked through -------------------------------------------
active = state.active_misconceptions()
if active:
    st.subheader("Worth revisiting")
    st.caption("Ideas we are still untangling together.")
    for misconception in active:
        with st.container(border=True):
            st.markdown(f"**{readable(misconception.concept_id)}**")
            st.write(misconception.description)

resolved = [m for m in state.misconceptions if not m.active]
if resolved:
    st.subheader("Cleared up")
    for misconception in resolved:
        st.markdown(
            f":material/check_circle: **{readable(misconception.concept_id)}** — "
            f"{misconception.description}"
        )

if state.demonstrated_concepts:
    st.subheader("You've shown you know")
    st.markdown(" ".join(f":green-badge[{readable(c)}]" for c in state.demonstrated_concepts))

# --- recent shape of the conversation ---------------------------------------
if state.previous_actions:
    st.subheader("How we've been working")
    recent = [action_label(a.value) for a in state.previous_actions[-8:]]
    counts: dict[str, int] = {}
    for label in recent:
        counts[label] = counts.get(label, 0) + 1
    st.markdown(
        " ".join(
            f":blue-badge[{label}{'' if count == 1 else f' ×{count}'}]"
            for label, count in counts.items()
        )
    )
