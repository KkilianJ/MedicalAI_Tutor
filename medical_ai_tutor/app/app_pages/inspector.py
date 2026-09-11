"""Developer view of the agent's internals.

This is the debug surface the architecture requires: learner state, last
diagnosis, chosen action, retrieval, both safety layers, revision count and
per-turn cost. It renders redacted traces only — protected solution text is
never displayed here or anywhere else in the UI.
"""

from __future__ import annotations

import streamlit as st

from medical_ai_tutor.app.ui import get_tutor, learner_state
from medical_ai_tutor.state.models import SafetyVerdict

VERDICT_COLOR = {
    SafetyVerdict.PASS.value: "green",
    SafetyVerdict.REVISE.value: "orange",
    SafetyVerdict.BLOCK.value: "red",
}
STAGE_LABELS = {
    "observe": "Observe",
    "diagnose": "Diagnose learner",
    "update_state": "Update learner model",
    "decide": "Choose pedagogical action",
    "tools": "Tools",
    "generate": "Generate response",
    "safety": "Safety gate",
    "persist": "Persist state and trace",
}

tutor = get_tutor()
state = learner_state()

st.title("Inspector")
st.caption(
    "The agent's internal state. Traces are redacted — a pre-safety candidate is "
    "never shown."
)

config = st.container(horizontal=True)
config.metric("Provider", tutor.client.name)
config.metric("Retrieval", tutor.retriever.mode)
config.metric("Passages", len(tutor.retriever.chunks))
config.metric("Protected", len(tutor.protected))

if state is None or not st.session_state.messages:
    st.caption("No turns yet in this session.")
    st.stop()

trace = next(
    (m.get("trace") for m in reversed(st.session_state.messages) if m.get("trace")), None
)

learner, turn = st.tabs(["Learner state", "Last turn"])

with learner:
    st.write(
        f"**Topic** `{state.current_topic or '—'}` · "
        f"**Exercise** `{state.current_exercise_id or '—'}` · "
        f"**Hint level** {state.hint_level}/{tutor.updater.hint_max} · "
        f"**Attempts** {state.attempt_count} · **Turns** {state.turn_count}"
    )
    if state.concept_mastery:
        st.dataframe(
            [
                {
                    "Concept": cid,
                    "Mastery": round(cm.score, 4),
                    "Confidence": round(cm.confidence, 3),
                    "Evidence": cm.evidence_count,
                    "Last turn": cm.last_updated_turn,
                }
                for cid, cm in sorted(
                    state.concept_mastery.items(), key=lambda kv: -kv[1].score
                )
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "Mastery": st.column_config.ProgressColumn(
                    "Mastery", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )
    if state.misconceptions:
        st.dataframe(
            [
                {
                    "Concept": m.concept_id,
                    "Active": m.active,
                    "Confidence": round(m.confidence, 3),
                    "First seen": m.first_seen_turn,
                    "Last seen": m.last_seen_turn,
                    "Description": m.description,
                }
                for m in state.misconceptions
            ],
            hide_index=True,
            width="stretch",
        )
    st.caption(f"Revealed units: {state.revealed_information_units or '—'}")
    if state.history_summary:
        with st.expander("Rolling history summary", icon=":material/history:"):
            st.write(state.history_summary)

with turn:
    if trace is None:
        st.caption("No trace captured.")
        st.stop()

    if trace.diagnosis:
        st.write(
            f"**Diagnosis** — intent `{trace.diagnosis.intent.value}`, quality "
            f"`{trace.diagnosis.response_quality.value}`, confidence "
            f"{trace.diagnosis.confidence:.2f}"
        )
    if trace.decision:
        st.write(
            f"**Decision** — `{trace.decision.action.value}` "
            f"(`{trace.decision.reason_code}`, confidence {trace.decision.confidence:.2f}), "
            f"target `{trace.decision.target_concept}`, "
            f"hint level {trace.decision.desired_hint_level}"
        )

    st.write(
        f"**Retrieval** — {'called' if trace.retrieved_section_ids else 'not called'}"
        + (f" · {', '.join(trace.retrieved_section_ids)}" if trace.retrieved_section_ids else "")
    )

    verdict = trace.safety_verdict.value
    st.write("**Safety**")
    st.badge(
        f"Semantic judge: {verdict}",
        color=VERDICT_COLOR.get(verdict, "grey"),
        icon=":material/gavel:",
    )
    if trace.deterministic:
        finding = trace.deterministic
        st.caption(
            f"Deterministic: leaked={finding.leaked} · n-gram {finding.ngram_overlap:.3f} · "
            f"longest run {finding.max_verbatim_run} · units {len(finding.covered_units)}"
            + (f" · {', '.join(finding.reason_codes)}" if finding.reason_codes else "")
        )
    st.caption(f"Revisions: {len(trace.revisions)} · fallback: {trace.used_fallback}")

    metrics = trace.metrics
    st.caption(
        f"{metrics.llm_calls} LLM calls · {metrics.tool_calls} tool calls · "
        f"{metrics.input_tokens}→{metrics.output_tokens} tokens · "
        f"{metrics.total_latency_ms:.0f} ms"
    )

    with st.status("Pipeline stages", type="compact", expanded=True):
        for stage in trace.stages:
            with st.status(
                f"{STAGE_LABELS.get(stage.name, stage.name)} · {stage.duration_ms:.1f} ms",
                type="step",
                state="error" if not stage.ok else "complete",
            ):
                if stage.error:
                    st.write(f":red[{stage.error}]")
                for key, value in (stage.detail or {}).items():
                    st.write(f"**{key.replace('_', ' ')}:** {value}")
