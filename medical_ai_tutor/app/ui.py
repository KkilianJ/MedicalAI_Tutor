"""Shared helpers for the Streamlit app.

Keeps the page scripts declarative. Nothing here talks to Streamlit except the
cached resource and the session bootstrap; the rest is plain translation from
the agent's internal vocabulary into something a learner should see.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..config import load_config
from ..state.models import LearnerState, PedagogicalAction
from ..tutor.orchestrator import build_orchestrator

# --------------------------------------------------------------------------- #
# Vocabulary translation
# --------------------------------------------------------------------------- #

# The agent's action names are engineering vocabulary. A learner should see what
# the tutor is *doing for them*, not the enum that produced it.
ACTION_LABEL: dict[str, str] = {
    PedagogicalAction.ASK_DIAGNOSTIC.value: "Checking where you are",
    PedagogicalAction.ASK_SOCRATIC.value: "Something to think through",
    PedagogicalAction.GIVE_HINT.value: "A hint",
    PedagogicalAction.GIVE_EXAMPLE.value: "An example",
    PedagogicalAction.EXPLAIN_CONCEPT.value: "An explanation",
    PedagogicalAction.CORRECT_MISCONCEPTION.value: "Clearing something up",
    PedagogicalAction.CHALLENGE.value: "A challenge",
    PedagogicalAction.QUIZ.value: "A quick check",
    PedagogicalAction.RETRIEVE.value: "From the textbook",
    PedagogicalAction.REDIRECT.value: "Back to the material",
    PedagogicalAction.REFUSE_SOLUTION.value: "Let's work it out",
}

ACTION_ICON: dict[str, str] = {
    PedagogicalAction.ASK_DIAGNOSTIC.value: ":material/help:",
    PedagogicalAction.ASK_SOCRATIC.value: ":material/psychology:",
    PedagogicalAction.GIVE_HINT.value: ":material/lightbulb:",
    PedagogicalAction.GIVE_EXAMPLE.value: ":material/menu_book:",
    PedagogicalAction.EXPLAIN_CONCEPT.value: ":material/school:",
    PedagogicalAction.CORRECT_MISCONCEPTION.value: ":material/change_circle:",
    PedagogicalAction.CHALLENGE.value: ":material/trending_up:",
    PedagogicalAction.QUIZ.value: ":material/quiz:",
    PedagogicalAction.RETRIEVE.value: ":material/search:",
    PedagogicalAction.REDIRECT.value: ":material/u_turn_left:",
    PedagogicalAction.REFUSE_SOLUTION.value: ":material/construction:",
}

# Qualitative bands. A learner reading "mastery 0.16" learns nothing useful and
# feels judged; "Building" tells them where they are and that it moves.
MASTERY_BANDS: list[tuple[float, str, str]] = [
    (0.25, "Just started", "grey"),
    (0.50, "Building", "orange"),
    (0.70, "Getting there", "blue"),
    (1.01, "Solid", "green"),
]


def mastery_band(score: float) -> tuple[str, str]:
    for ceiling, label, colour in MASTERY_BANDS:
        if score < ceiling:
            return label, colour
    return MASTERY_BANDS[-1][1], MASTERY_BANDS[-1][2]


def readable(concept_id: str | None) -> str:
    if not concept_id:
        return "—"
    return concept_id.replace("_", " ").replace("3lgm2", "3LGM²").strip().capitalize()


def action_label(action: str) -> str:
    return ACTION_LABEL.get(action, action.replace("_", " ").capitalize())


def action_icon(action: str) -> str:
    return ACTION_ICON.get(action, ":material/chat:")


# Conversation starters for a cold chat. Keys are the pill labels (with icon
# markup), values the message actually sent. Kept here so tests can import them
# without executing a page script.
STARTERS: dict[str, str] = {
    ":blue[:material/psychology:] Explain a concept": (
        "What is interoperability in a health information system?"
    ),
    ":green[:material/edit_note:] Check my thinking": (
        "I think normalization just means splitting a large table into smaller tables."
    ),
    ":violet[:material/travel_explore:] Look it up": (
        "According to the textbook, what does the glossary say about interoperability?"
    ),
    ":orange[:material/help:] I'm stuck": (
        "I don't really understand what a communication server is for."
    ),
}


def starter_labelled(fragment: str) -> str:
    """The full pill label containing `fragment`. Used by the UI tests."""
    for label in STARTERS:
        if fragment.lower() in label.lower():
            return label
    raise KeyError(f"no starter matching {fragment!r}")


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Starting the tutor…")
def get_tutor():
    """The orchestrator is expensive to build (corpus + index), so cache it."""
    return build_orchestrator(load_config())


def init_session_state() -> None:
    """All session-state initialisation, in one place, shared across pages."""
    defaults: dict[str, Any] = {
        "session_id": "learner",
        "messages": [],
        "exercise_id": None,
        "show_reasoning": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def learner_state() -> LearnerState | None:
    return get_tutor().get_state(st.session_state.session_id)


def hydrate_transcript() -> None:
    """Rebuild the visible chat from persisted state.

    Learner state lives in SQLite and survives a page reload, but
    `st.session_state.messages` does not. Without this the chat looks empty
    while the tutor still remembers the conversation — so it answers in a
    context the learner cannot see.
    """
    if st.session_state.messages:
        return
    state = learner_state()
    if state is None or not state.dialogue:
        return
    st.session_state.messages = [
        {
            "role": "user" if turn.role.value == "student" else "assistant",
            "content": turn.content,
            "action": turn.action.value if turn.action else "",
            "citations": [],
            "sections": [],
            "trace": None,
            "restored": True,
        }
        for turn in state.dialogue
    ]


def send(message: str) -> None:
    """Run one turn and record it in the transcript."""
    tutor = get_tutor()
    st.session_state.messages.append({"role": "user", "content": message})
    response, trace = tutor.run_turn(
        st.session_state.session_id, message, st.session_state.exercise_id
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.text,
            "action": response.action.value,
            "citations": response.citations,
            "sections": response.retrieved_section_ids,
            # Redacted: a raw trace can hold a pre-safety candidate.
            "trace": trace.redacted(),
            "failures": trace.failed_stages(),
        }
    )


def reset_session() -> None:
    get_tutor().reset_session(st.session_state.session_id)
    st.session_state.messages = []
    # Drop the starter pill's selection too: an empty transcript re-renders the
    # pills, and a retained selection would immediately re-send that starter.
    st.session_state.pop("starter", None)


def last_turn_failures() -> list[tuple[str, str]]:
    """Provider failures from the most recent turn, if any."""
    for message in reversed(st.session_state.messages):
        if message.get("role") == "assistant":
            return list(message.get("failures") or [])
    return []


def config_problems() -> list[str]:
    """Configuration warnings worth showing (bad model/provider pairing, etc.)."""
    return list(load_config().notes)


def active_exercise():
    if not st.session_state.exercise_id:
        return None
    return get_tutor().exercises.get(st.session_state.exercise_id)


def exercises_by_chapter() -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for exercise in get_tutor().exercises.all():
        grouped.setdefault(exercise.chapter or "?", []).append(exercise)
    return dict(sorted(grouped.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 99))


CHAPTER_TITLES = {
    "1": "Introduction",
    "2": "Basic concepts and terms",
    "3": "Architecture of health information systems",
    "4": "Managing health information systems",
    "5": "Quality and evaluation",
    "6": "Information systems for specific settings",
}


def chapter_title(number: str) -> str:
    return CHAPTER_TITLES.get(number, f"Chapter {number}")


# --------------------------------------------------------------------------- #
# Shared chrome
# --------------------------------------------------------------------------- #


def render_sidebar() -> None:
    """Learner context: what you're working on, how it's going."""
    tutor = get_tutor()
    state = learner_state()

    with st.sidebar:
        exercise = active_exercise()
        if exercise:
            st.caption("Working on")
            st.markdown(f"**{exercise.title}**")
            st.caption(f"Exercise {exercise.exercise_id}")
            if st.button(
                "Put this aside", icon=":material/close:", width="stretch", type="tertiary"
            ):
                st.session_state.exercise_id = None
                st.rerun()
        else:
            st.caption("No exercise selected")
            st.caption("Pick one from **Exercises**, or just ask a question.")

        if state and state.concept_mastery:
            st.caption("Your concepts")
            top = sorted(state.concept_mastery.items(), key=lambda kv: -kv[1].score)[:4]
            for concept_id, mastery in top:
                label, colour = mastery_band(mastery.score)
                st.markdown(
                    f"{readable(concept_id)} &nbsp;:{colour}-badge[{label}]",
                    unsafe_allow_html=False,
                )
                st.progress(min(1.0, mastery.score))

        if st.button("Start over", icon=":material/restart_alt:", width="stretch"):
            reset_session()
            st.rerun()

        st.caption(
            f"{tutor.client.name} · {len(tutor.retriever.chunks)} passages · "
            f"{len(tutor.exercises)} exercises"
        )
        if tutor.provider_warning:
            st.caption(f":orange[{tutor.provider_warning}]")
