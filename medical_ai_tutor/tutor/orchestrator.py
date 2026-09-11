"""The agent loop.

One obvious `run_turn()` path, with every transition recorded in a `TurnTrace`:

    student message
      -> observation
      -> diagnosis                (LLM, structured)
      -> learner-state update     (deterministic)
      -> pedagogical decision     (LLM, structured, runtime-enforced)
      -> optional tool call(s)    (runtime-executed, budgeted)
      -> tutor generation         (LLM, free text)
      -> deterministic safety gate
      -> independent semantic judge
      -> PASS / REVISE / BLOCK
      -> bounded critique-guided revision, else deterministic safe fallback
      -> persist state + trace

Retrieval is a conditional tool inside this loop, not a mandatory stage.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..config import Config, get_config
from ..llm.base import LLMClient, UsageAccumulator
from ..llm.factory import build_client_or_mock
from ..retrieval.hybrid import HybridRetriever
from ..safety.gate import SafetyGate
from ..safety.protected_store import ProtectedStore
from ..state.models import (
    ExerciseMeta,
    LearnerState,
    Observation,
    PedagogicalAction,
    PedagogicalDecision,
    RetrievedPassage,
    SafetyVerdict,
    TurnMetrics,
    TutorResponse,
)
from ..state.store import SessionStore
from ..state.updater import StateUpdater
from ..tools.exercise_tool import ExerciseCatalog, GetExerciseTool
from ..tools.registry import ToolRegistry
from ..tools.retrieval_tool import SearchCourseMaterialTool
from ..tracing.models import StageRecord, ToolCallRecord, TurnTrace
from ..tracing.store import TraceStore
from .diagnosis import LearnerDiagnoser
from .fallback import safe_fallback_text
from .generator import TutorGenerator
from .policy import PedagogicalPolicy
from .revision import RevisionLoop

RETRIEVAL_TOOL = "search_course_material"


class Orchestrator:
    """Wires the components together and runs one turn at a time."""

    def __init__(
        self,
        config: Config,
        client: LLMClient,
        session_store: SessionStore,
        trace_store: TraceStore,
        retriever: HybridRetriever,
        exercise_catalog: ExerciseCatalog,
        protected_store: ProtectedStore,
        provider_warning: str | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.sessions = session_store
        self.traces = trace_store
        self.retriever = retriever
        self.exercises = exercise_catalog
        self.protected = protected_store
        self.provider_warning = provider_warning

        self.max_tool_calls = int(config.get("limits.max_tool_calls_per_turn", 3))
        self.max_decision_steps = int(config.get("limits.max_decision_steps_per_turn", 4))
        self.retrieval_top_k = int(config.get("retrieval.top_k", 4))
        self.trace_enabled = bool(config.get("tracing.enabled", True))

        self.tools = ToolRegistry(max_calls_per_turn=self.max_tool_calls)
        self.tools.register(
            SearchCourseMaterialTool(
                retriever, max_passage_chars=int(config.get("context.max_passage_chars", 1100))
            )
        )
        self.tools.register(GetExerciseTool(exercise_catalog))

        self.updater = StateUpdater(config)
        self.gate = SafetyGate(config, protected_store, client)
        known_concepts = sorted(
            {c for meta in exercise_catalog.all() for c in meta.concepts}
        )
        self.diagnoser = LearnerDiagnoser(client, config, known_concepts=known_concepts)
        self.policy = PedagogicalPolicy(client, config)
        self.generator = TutorGenerator(client, config)
        self.reviser = RevisionLoop(client, config, self.gate)

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #
    def create_session(self, session_id: str | None = None, topic: str | None = None) -> LearnerState:
        return self.sessions.create(session_id, topic)

    def get_state(self, session_id: str) -> LearnerState | None:
        return self.sessions.load(session_id)

    def reset_session(self, session_id: str) -> LearnerState:
        return self.sessions.reset(session_id)

    # ------------------------------------------------------------------ #
    # Turn
    # ------------------------------------------------------------------ #
    def run_turn(
        self,
        session_id: str,
        message: str,
        exercise_id: str | None = None,
    ) -> tuple[TutorResponse, TurnTrace]:
        started = time.perf_counter()
        usage = UsageAccumulator()
        metrics = TurnMetrics()

        state_before = self.sessions.load_or_create(session_id)
        turn_id = state_before.turn_count + 1
        trace = TurnTrace(
            trace_id=f"{session_id}:{turn_id}:{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            turn_id=turn_id,
            student_message=message,
        )

        @contextmanager
        def stage(name: str) -> Iterator[StageRecord]:
            record = StageRecord(name=name)
            stage_started = time.perf_counter()
            try:
                yield record
            except Exception as exc:
                record.ok = False
                record.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record.duration_ms = (time.perf_counter() - stage_started) * 1000
                metrics.stage_latency_ms[name] = round(record.duration_ms, 2)
                trace.stages.append(record)

        # -- 1. Observe ------------------------------------------------- #
        with stage("observe") as record:
            observation = Observation.from_student(message, turn_id=turn_id)
            trace.observations.append(observation)
            if exercise_id:
                # An explicitly selected exercise wins over inference.
                state_before = state_before.model_copy(
                    update={"current_exercise_id": exercise_id}
                )
            exercise = self._active_exercise(state_before, exercise_id)
            protected = self.protected.has(exercise.exercise_id if exercise else None)
            record.detail = {
                "exercise_id": exercise.exercise_id if exercise else None,
                "protected": protected,
                "message_chars": len(message),
            }

        # -- 2. Diagnose ------------------------------------------------- #
        with stage("diagnose") as record:
            diagnosis, in_tokens, out_tokens, error = self.diagnoser.run(
                observation, state_before, exercise, protected
            )
            usage.llm_calls += 1
            usage.input_tokens += in_tokens
            usage.output_tokens += out_tokens
            trace.diagnosis = diagnosis
            record.ok = error is None
            record.error = error
            record.detail = {
                "intent": diagnosis.intent.value,
                "response_quality": diagnosis.response_quality.value,
                "evidence": len(diagnosis.mastery_evidence),
                "misconceptions": len(diagnosis.detected_misconceptions),
            }

        # -- 3. Update learner state -------------------------------------- #
        with stage("update_state") as record:
            state, delta = self.updater.apply(state_before, diagnosis, observation)
            trace.state_delta = delta
            # Re-resolve: diagnosis may have switched the active exercise.
            exercise = self._active_exercise(state, None)
            protected = self.protected.has(exercise.exercise_id if exercise else None)
            record.detail = {
                "mastery_changes": len(delta.mastery_changes),
                "misconceptions_added": delta.misconceptions_added,
                "misconceptions_resolved": delta.misconceptions_resolved,
                "hint_level": state.hint_level,
                "attempt_count": state.attempt_count,
            }

        # -- 4. Decide (+ conditional retrieval) --------------------------- #
        passages: list[RetrievedPassage] = []
        self.tools.begin_turn()
        decision: PedagogicalDecision | None = None

        with stage("decide") as record:
            steps = 0
            has_context = False
            while steps < self.max_decision_steps:
                steps += 1
                decision, in_tokens, out_tokens, error = self.policy.select(
                    state=state,
                    diagnosis=diagnosis,
                    student_message=message,
                    exercise=exercise,
                    protected=protected,
                    tool_specs=self.tools.specs(),
                    has_retrieved_context=has_context,
                )
                usage.llm_calls += 1
                usage.input_tokens += in_tokens
                usage.output_tokens += out_tokens
                if error:
                    record.ok = False
                    record.error = error

                needs_more = decision.needs_retrieval or (
                    decision.action == PedagogicalAction.RETRIEVE
                )
                if not needs_more or self.tools.budget_remaining == 0:
                    break

                # Retrieval runs here, then the policy decides again with the
                # material in hand. This is what MAX_DECISION_STEPS bounds.
                new_passages = self._retrieve(decision, trace, metrics)
                passages.extend(p for p in new_passages if p not in passages)
                has_context = True

            trace.decision = decision
            record.detail = {
                "action": decision.action.value if decision else None,
                "reason_code": decision.reason_code if decision else None,
                "decision_steps": steps,
                "retrieved": len(passages),
            }

        assert decision is not None  # the loop always assigns at least once

        with stage("tools") as record:
            trace.retrieved_section_ids = [p.section_id or p.chunk_id for p in passages]
            metrics.tool_calls = self.tools.calls_this_turn
            record.detail = {
                "tool_calls": self.tools.calls_this_turn,
                "retrieval_used": bool(passages),
                "sections": trace.retrieved_section_ids,
                "mode": self.retriever.mode,
            }

        # -- 5. Generate ---------------------------------------------------- #
        with stage("generate") as record:
            candidate, in_tokens, out_tokens, error = self.generator.generate(
                state=state,
                diagnosis=diagnosis,
                decision=decision,
                student_message=message,
                passages=passages,
                exercise=exercise,
                protected=protected,
            )
            usage.llm_calls += 1
            usage.input_tokens += in_tokens
            usage.output_tokens += out_tokens
            used_fallback = False
            if error or not candidate:
                candidate = safe_fallback_text(decision, state)
                used_fallback = True
            trace.candidate_response = candidate
            record.ok = error is None
            record.error = error
            record.detail = {"chars": len(candidate), "generation_fallback": used_fallback}

        # -- 6. Safety ------------------------------------------------------ #
        with stage("safety") as record:
            report = self.gate.evaluate(candidate, state, exercise)
            judge_in, judge_out = self.gate.last_judge_tokens
            if report.judge is not None:
                usage.llm_calls += 1
                usage.input_tokens += judge_in
                usage.output_tokens += judge_out

            final_text = candidate
            revision_records: list[Any] = []

            if report.verdict == SafetyVerdict.REVISE:
                (
                    final_text,
                    report,
                    revision_records,
                    revision_fallback,
                    in_tokens,
                    out_tokens,
                ) = self.reviser.run(
                    candidate=candidate,
                    report=report,
                    state=state,
                    diagnosis=diagnosis,
                    decision=decision,
                    passages=passages,
                    exercise=exercise,
                )
                usage.llm_calls += len(revision_records)
                usage.input_tokens += in_tokens
                usage.output_tokens += out_tokens
                used_fallback = used_fallback or revision_fallback
            elif report.verdict == SafetyVerdict.BLOCK:
                # A block is not revised: emit the deterministic safe fallback
                # and re-verify it before it can reach the learner.
                final_text = safe_fallback_text(decision, state)
                report = self.gate.evaluate(final_text, state, exercise)
                used_fallback = True

            trace.deterministic = report.deterministic
            trace.judge = report.judge
            trace.safety_verdict = report.verdict
            trace.revisions = revision_records
            trace.used_fallback = used_fallback
            trace.final_response = final_text

            if report.verdict != SafetyVerdict.PASS:
                # Last-resort guarantee: nothing that still fails review is emitted.
                final_text = safe_fallback_text(None, state)
                trace.final_response = final_text
                trace.used_fallback = used_fallback = True
                report = report.model_copy(update={"verdict": SafetyVerdict.PASS})

            trace.observations.append(
                Observation.from_safety(
                    {
                        "verdict": report.verdict.value,
                        "reason_code": report.reason_code,
                        "revisions": len(revision_records),
                        "used_fallback": used_fallback,
                    },
                    turn_id=turn_id,
                )
            )
            record.detail = {
                "verdict": report.verdict.value,
                "reason_code": report.reason_code,
                "protected_active": report.protected_exercise_active,
                "revisions": len(revision_records),
                "used_fallback": used_fallback,
            }

        # -- 7. Persist ------------------------------------------------------ #
        with stage("persist") as record:
            final_state = self.updater.record_action_and_response(
                state=state,
                decision=decision,
                response_text=final_text,
                turn_id=turn_id,
                delta=delta,
                revealed_units=self.generator.disclosure_units(decision),
            )
            self.sessions.save(final_state)

            metrics.llm_calls = usage.llm_calls
            metrics.input_tokens = usage.input_tokens
            metrics.output_tokens = usage.output_tokens
            metrics.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            trace.metrics = metrics
            if self.trace_enabled:
                self.traces.save(trace)
            record.detail = {
                "hint_level": final_state.hint_level,
                "turn_count": final_state.turn_count,
                "trace_stored": self.trace_enabled,
            }

        response = TutorResponse(
            session_id=session_id,
            turn_id=turn_id,
            text=final_text,
            action=decision.action,
            citations=[p.citation() for p in passages],
            retrieved_section_ids=trace.retrieved_section_ids,
            safety_verdict=report.verdict,
            revision_count=len(trace.revisions),
            used_fallback=trace.used_fallback,
        )
        return response, trace

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _active_exercise(
        self, state: LearnerState, explicit_id: str | None
    ) -> ExerciseMeta | None:
        exercise_id = explicit_id or state.current_exercise_id
        return self.exercises.get(exercise_id) if exercise_id else None

    def _retrieve(
        self, decision: PedagogicalDecision, trace: TurnTrace, metrics: TurnMetrics
    ) -> list[RetrievedPassage]:
        query = (decision.retrieval_query or decision.target_concept or "").strip()
        if not query:
            return []

        result = self.tools.execute(
            RETRIEVAL_TOOL, {"query": query, "top_k": self.retrieval_top_k}, role="tutor"
        )
        trace.tool_calls.append(
            ToolCallRecord(
                tool_name=RETRIEVAL_TOOL,
                arguments={"query": query[:160], "top_k": self.retrieval_top_k},
                ok=result.ok,
                result_summary=result.summary(),
                duration_ms=round(result.duration_ms, 2),
                error=result.error,
            )
        )
        trace.observations.append(result.to_observation(trace.turn_id))
        if not result.ok:
            return []

        return [
            RetrievedPassage(
                chunk_id=item["chunk_id"],
                doc_id="",
                section_id=item.get("section_id"),
                section_title=item.get("section_title"),
                page=item.get("page"),
                text=item.get("text", ""),
                score=float(item.get("score", 0.0)),
            )
            for item in result.payload.get("passages", [])
        ]

    def list_traces(self, session_id: str, limit: int = 50) -> list[TurnTrace]:
        return self.traces.list_for_session(session_id, limit)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def build_orchestrator(
    config: Config | None = None,
    client: LLMClient | None = None,
    db_path: str | None = None,
) -> Orchestrator:
    """Wire an orchestrator from configuration.

    Falls back to the mock provider when no API key is configured, so the app,
    the tests and the evaluation suite all start without credentials.
    """
    config = config or get_config()
    warning: str | None = None
    if client is None:
        client, warning = build_client_or_mock(config)

    database = db_path or str(config.path("storage.db_path"))
    session_store = SessionStore(database)
    trace_store = TraceStore(database, connection=session_store.connection)

    retriever = HybridRetriever.from_corpus_file(
        config.path("storage.searchable_dir") / "corpus.jsonl",
        bm25_weight=float(config.get("retrieval.bm25_weight", 0.6)),
        embedding_weight=float(config.get("retrieval.embedding_weight", 0.4)),
        dedup_threshold=float(config.get("retrieval.dedup_jaccard_threshold", 0.85)),
        use_embeddings=bool(config.get("retrieval.use_embeddings", False)),
        embedding_model=str(config.get("retrieval.embedding_model", "")),
        min_score=float(config.get("retrieval.min_score", 0.0)),
    )
    catalog = ExerciseCatalog.from_file(config.path("storage.exercises_dir") / "exercises.json")
    protected = ProtectedStore.from_file(config.path("storage.protected_dir") / "solutions.json")

    return Orchestrator(
        config=config,
        client=client,
        session_store=session_store,
        trace_store=trace_store,
        retriever=retriever,
        exercise_catalog=catalog,
        protected_store=protected,
        provider_warning=warning,
    )
