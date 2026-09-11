"""Browse the textbook's exercises and pick one to work on."""

from __future__ import annotations

import streamlit as st

from medical_ai_tutor.app.ui import (
    chapter_title,
    exercises_by_chapter,
    get_tutor,
    readable,
)

tutor = get_tutor()
grouped = exercises_by_chapter()

st.title("Exercises")
st.caption(
    f"{len(tutor.exercises)} exercises from the textbook. Pick one and I'll help you "
    "work through it — the official answers stay closed."
)

if not grouped:
    st.warning(
        "No exercises loaded. Run `make ingest` to parse the textbook.",
        icon=":material/inventory_2:",
    )
    st.stop()

chapters = list(grouped)
choice = st.segmented_control(
    "Chapter",
    ["All", *chapters],
    default="All",
    label_visibility="collapsed",
)
visible = chapters if choice in (None, "All") else [choice]

for chapter in visible:
    st.subheader(f"{chapter}. {chapter_title(chapter)}")
    for exercise in grouped[chapter]:
        selected = st.session_state.exercise_id == exercise.exercise_id
        with st.container(border=True):
            header = st.container(horizontal=True, vertical_alignment="center")
            header.markdown(f"**{exercise.title}**")
            if selected:
                header.badge("Working on this", icon=":material/check:", color="green")

            st.caption(f"Exercise {exercise.exercise_id}")
            st.write(
                exercise.question[:280] + ("…" if len(exercise.question) > 280 else "")
            )
            if exercise.concepts:
                st.caption("Concepts: " + " · ".join(readable(c) for c in exercise.concepts))

            actions = st.container(horizontal=True)
            if actions.button(
                "Work on this" if not selected else "Continue",
                key=f"pick-{exercise.exercise_id}",
                icon=":material/play_arrow:",
                type="primary" if not selected else "secondary",
            ):
                st.session_state.exercise_id = exercise.exercise_id
                st.switch_page("app_pages/learn.py")
            if len(exercise.question) > 280:
                with actions.popover("Full question", icon=":material/article:"):
                    st.write(exercise.question)
