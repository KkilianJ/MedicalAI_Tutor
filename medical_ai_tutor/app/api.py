"""FastAPI backend.

Public endpoints never return protected solution text: traces are served through
`TurnTrace.redacted()` and no route reads `data/protected/`.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import get_config
from ..retrieval.ingest import run_ingestion
from ..state.models import LearnerState, TutorResponse
from ..tutor.orchestrator import Orchestrator, build_orchestrator


class CreateSessionRequest(BaseModel):
    session_id: str | None = None
    topic: str | None = None


class TurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    exercise_id: str | None = None


class IngestRequest(BaseModel):
    pdf_path: str | None = None
    doc_id: str = "winter2023_his"


class HealthResponse(BaseModel):
    status: str
    provider: str
    retrieval_mode: str
    corpus_chunks: int
    exercises: int
    protected_solutions: int
    provider_warning: str | None = None


def create_app(orchestrator: Orchestrator | None = None) -> FastAPI:
    config = get_config()
    runtime = orchestrator or build_orchestrator(config)

    app = FastAPI(
        title="Medical AI Tutor",
        version="0.1.0",
        description=(
            "Agentic tutor for Medical Informatics / Health Information Systems. "
            "Protected exercise solutions are never returned by any endpoint."
        ),
    )
    app.state.orchestrator = runtime

    def _state_or_404(session_id: str) -> LearnerState:
        state = runtime.get_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
        return state

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            provider=runtime.client.name,
            retrieval_mode=runtime.retriever.mode,
            corpus_chunks=len(runtime.retriever.chunks),
            exercises=len(runtime.exercises),
            protected_solutions=len(runtime.protected),
            provider_warning=runtime.provider_warning,
        )

    @app.post("/sessions", status_code=201)
    def create_session(request: CreateSessionRequest) -> dict[str, Any]:
        state = runtime.create_session(request.session_id, request.topic)
        return {"session_id": state.session_id, "created_at": state.created_at.isoformat()}

    @app.post("/sessions/{session_id}/turn", response_model=TutorResponse)
    def run_turn(session_id: str, request: TurnRequest) -> TutorResponse:
        if request.exercise_id and not runtime.exercises.get(request.exercise_id):
            raise HTTPException(
                status_code=400, detail=f"unknown exercise {request.exercise_id!r}"
            )
        response, _ = runtime.run_turn(session_id, request.message, request.exercise_id)
        return response

    @app.get("/sessions/{session_id}/state", response_model=LearnerState)
    def get_state(session_id: str) -> LearnerState:
        return _state_or_404(session_id)

    @app.get("/sessions/{session_id}/traces")
    def get_traces(session_id: str, limit: int = 50) -> dict[str, Any]:
        _state_or_404(session_id)
        capped = min(limit, int(config.get("tracing.max_traces_returned", 50)))
        traces = runtime.list_traces(session_id, capped)
        # Redacted: a raw trace may contain a pre-safety candidate.
        return {
            "session_id": session_id,
            "count": len(traces),
            "traces": [t.redacted().model_dump(mode="json") for t in traces],
        }

    @app.delete("/sessions/{session_id}")
    def reset_session(session_id: str) -> dict[str, Any]:
        state = runtime.reset_session(session_id)
        return {"session_id": state.session_id, "reset": True}

    @app.get("/exercises")
    def list_exercises() -> dict[str, Any]:
        return {
            "count": len(runtime.exercises),
            "exercises": [
                {
                    "exercise_id": e.exercise_id,
                    "title": e.title,
                    "chapter": e.chapter,
                    "concepts": e.concepts,
                    "protected": e.protected_unit_count > 0,
                }
                for e in runtime.exercises.all()
            ],
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict[str, Any]:
        raw_dir = config.path("storage.raw_dir")
        pdf_path = request.pdf_path or str(raw_dir / f"{request.doc_id}.pdf")
        try:
            manifest = run_ingestion(
                pdf_path=pdf_path,
                searchable_dir=config.path("storage.searchable_dir"),
                exercises_dir=config.path("storage.exercises_dir"),
                protected_dir=config.path("storage.protected_dir"),
                chunk_config=config.get("retrieval.chunk", {}),
                doc_id=request.doc_id,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # The manifest reports counts and verification results only.
        return manifest

    return app


app = create_app()
