"""Medical AI Tutor — learner-facing Streamlit app.

Four pages: Learn (the tutor), Exercises (pick something to work on), Progress
(what you've covered), Inspector (the agent's internals, for developers).

The learner-facing pages deliberately hide the machinery. Nothing here renders
protected solution text: traces reach the UI already redacted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run medical_ai_tutor/app/streamlit_app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medical_ai_tutor.app.ui import init_session_state, render_sidebar  # noqa: E402

st.set_page_config(
    page_title="Medical AI Tutor",
    page_icon=":material/school:",
    layout="centered",  # a tutor is a reading experience, not a dashboard
    initial_sidebar_state="expanded",
)

init_session_state()

page = st.navigation(
    [
        st.Page("app_pages/learn.py", title="Learn", icon=":material/forum:", default=True),
        st.Page("app_pages/exercises.py", title="Exercises", icon=":material/task_alt:"),
        st.Page("app_pages/progress.py", title="Progress", icon=":material/insights:"),
        st.Page("app_pages/inspector.py", title="Inspector", icon=":material/build:"),
    ],
    position="top",
)

render_sidebar()
page.run()
