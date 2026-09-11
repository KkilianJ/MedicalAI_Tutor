"""The tutor conversation — the page a learner actually lives in."""

from __future__ import annotations

import streamlit as st

from medical_ai_tutor.app.ui import (
    STARTERS,
    action_icon,
    action_label,
    active_exercise,
    config_problems,
    get_tutor,
    hydrate_transcript,
    last_turn_failures,
    readable,
    send,
)

tutor = get_tutor()
exercise = active_exercise()
hydrate_transcript()

st.title("Let's work through it")
st.caption(
    "Ask anything about health information systems. I'll figure out where you are "
    "and take it from there — I won't just hand you answers."
)

# A provider that is failing must be visible. Without this the runtime's own
# fallbacks make a broken API key look like an unhelpful tutor.
for problem in config_problems():
    st.warning(problem, icon=":material/settings_alert:")

failures = last_turn_failures()
if failures:
    st.error(
        "The language model could not be reached on the last turn, so you are "
        "seeing a built-in fallback reply rather than real tutoring.\n\n"
        + "\n".join(f"- **{stage}** — {error}" for stage, error in failures),
        icon=":material/cloud_off:",
    )

if exercise:
    with st.container(border=True):
        st.markdown(f"**Exercise {exercise.exercise_id} — {exercise.title}**")
        st.write(exercise.question)
        if exercise.concepts:
            st.caption("Concepts: " + " · ".join(readable(c) for c in exercise.concepts))

# --- transcript -------------------------------------------------------------
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            st.write(message["content"])
        continue

    with st.chat_message("assistant", avatar=":material/school:"):
        action = message.get("action", "")
        if action:
            st.caption(f"{action_icon(action)} {action_label(action)}")
        st.write(message["content"])

        citations = message.get("citations") or []
        if citations:
            with st.expander(
                f"Where this comes from ({len(citations)})", icon=":material/menu_book:"
            ):
                for citation in citations:
                    st.markdown(f"- {citation}")

        if st.session_state.show_reasoning and message.get("trace") is not None:
            trace = message["trace"]
            with st.status("How I decided", type="compact"):
                if trace.diagnosis:
                    st.write(
                        f"**Read your message as:** {trace.diagnosis.intent.value.replace('_', ' ')} "
                        f"— {trace.diagnosis.response_quality.value.replace('_', ' ')}"
                    )
                if trace.decision:
                    st.write(f"**Chose:** {action_label(trace.decision.action.value)}")
                st.write(
                    "**Looked at the textbook:** "
                    + (", ".join(trace.retrieved_section_ids) if trace.retrieved_section_ids
                       else "no — this came from our conversation")
                )

# --- cold start -------------------------------------------------------------
# The starter is consumed in the same run it is picked. Setting a flag and
# calling st.rerun() here would loop forever: the transcript is still empty on
# the next run, so the pill would re-fire before the send below is ever reached.
picked = None
if not st.session_state.messages:
    st.caption("Not sure where to start?")
    picked = st.pills(
        "Starters", list(STARTERS), label_visibility="collapsed", key="starter"
    )

# --- input ------------------------------------------------------------------
typed = st.chat_input(
    "Ask a question, or tell me what you think…", submit_mode="disable"
)
prompt = typed or (STARTERS[picked] if picked else None)

if prompt:
    with st.spinner("Thinking it through…"):
        send(prompt)
    st.rerun()

with st.sidebar:
    st.toggle(
        "Show how I decided",
        key="show_reasoning",
        help="Adds a short note under each reply explaining the tutor's reasoning.",
    )
