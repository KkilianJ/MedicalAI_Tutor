"""Core domain models.

Everything that crosses a component boundary in this system is one of these
Pydantic models. Validation is a hard runtime guarantee: an LLM cannot invent an
action name, push a mastery score out of [0, 1], or smuggle an unbounded hint
level past the schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ObservationSource(str, Enum):
    STUDENT = "student"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    SAFETY = "safety"
    RUNTIME = "runtime"


class ObservationKind(str, Enum):
    MESSAGE = "message"
    PASSAGE = "passage"
    RESULT = "result"
    VERDICT = "verdict"
    ERROR = "error"


class Intent(str, Enum):
    CONCEPT_QUESTION = "concept_question"
    EXERCISE_HELP = "exercise_help"
    STUDENT_ATTEMPT = "student_attempt"
    DIRECT_ANSWER_REQUEST = "direct_answer_request"
    CLARIFICATION_REQUEST = "clarification_request"
    ANSWER_TO_TUTOR_QUESTION = "answer_to_tutor_question"
    OFF_TOPIC = "off_topic"
    PROMPT_INJECTION = "prompt_injection"
    UNKNOWN = "unknown"


class ResponseQuality(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


class PedagogicalAction(str, Enum):
    """The complete, closed set of tutor actions. Exactly one per decision."""

    ASK_DIAGNOSTIC = "ASK_DIAGNOSTIC"
    ASK_SOCRATIC = "ASK_SOCRATIC"
    GIVE_HINT = "GIVE_HINT"
    GIVE_EXAMPLE = "GIVE_EXAMPLE"
    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"
    CORRECT_MISCONCEPTION = "CORRECT_MISCONCEPTION"
    CHALLENGE = "CHALLENGE"
    QUIZ = "QUIZ"
    RETRIEVE = "RETRIEVE"
    REDIRECT = "REDIRECT"
    REFUSE_SOLUTION = "REFUSE_SOLUTION"


class SafetyVerdict(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    BLOCK = "BLOCK"


class Role(str, Enum):
    STUDENT = "student"
    TUTOR = "tutor"


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


class Observation(BaseModel):
    """A single thing the agent perceived during a turn.

    Student messages, tool results, and safety verdicts are all observations, so
    the trace of a turn is a homogeneous, inspectable sequence.
    """

    model_config = ConfigDict(use_enum_values=False)

    source: ObservationSource
    kind: ObservationKind
    content: dict[str, Any] = Field(default_factory=dict)
    turn_id: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=_utcnow)

    @classmethod
    def from_student(cls, message: str, turn_id: int) -> Observation:
        return cls(
            source=ObservationSource.STUDENT,
            kind=ObservationKind.MESSAGE,
            content={"message": message},
            turn_id=turn_id,
        )

    @classmethod
    def from_tool(
        cls, tool_name: str, payload: dict[str, Any], turn_id: int, ok: bool = True
    ) -> Observation:
        return cls(
            source=ObservationSource.TOOL,
            kind=ObservationKind.RESULT if ok else ObservationKind.ERROR,
            content={"tool": tool_name, **payload},
            turn_id=turn_id,
        )

    @classmethod
    def from_safety(cls, payload: dict[str, Any], turn_id: int) -> Observation:
        return cls(
            source=ObservationSource.SAFETY,
            kind=ObservationKind.VERDICT,
            content=payload,
            turn_id=turn_id,
        )

    def short(self) -> str:
        if self.kind == ObservationKind.MESSAGE:
            return str(self.content.get("message", ""))[:400]
        return str(self.content)[:400]


# --------------------------------------------------------------------------- #
# Learner state components
# --------------------------------------------------------------------------- #


class ConceptMastery(BaseModel):
    """Estimated mastery of one concept. An estimate, never ground truth."""

    concept_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_count: int = Field(ge=0, default=0)
    last_updated_turn: int = Field(ge=0, default=0)


class Misconception(BaseModel):
    concept_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    active: bool = True
    first_seen_turn: int = Field(ge=0, default=0)
    last_seen_turn: int = Field(ge=0, default=0)


class DialogueTurn(BaseModel):
    role: Role
    content: str
    turn_id: int = Field(ge=0)
    action: PedagogicalAction | None = None
    timestamp: datetime = Field(default_factory=_utcnow)


class HintRecord(BaseModel):
    """A hint already given, so the tutor never repeats or over-reveals."""

    exercise_id: str | None = None
    hint_level: int = Field(ge=0)
    text: str
    turn_id: int = Field(ge=0)


class LearnerState(BaseModel):
    """Durable, structured estimate of the learner. Persisted between turns."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: str = Field(min_length=1)
    current_topic: str | None = None
    current_exercise_id: str | None = None

    concept_mastery: dict[str, ConceptMastery] = Field(default_factory=dict)
    misconceptions: list[Misconception] = Field(default_factory=list)

    uncertain_concepts: list[str] = Field(default_factory=list)
    demonstrated_concepts: list[str] = Field(default_factory=list)

    hint_level: int = Field(ge=0, default=0)
    attempt_count: int = Field(ge=0, default=0)

    previous_actions: list[PedagogicalAction] = Field(default_factory=list)
    previous_hints: list[HintRecord] = Field(default_factory=list)
    # Coarse labels for what has already been disclosed about the active
    # exercise. Used by the policy and by prompts, never to store solution text.
    revealed_information_units: list[str] = Field(default_factory=list)

    latest_intent: Intent | None = None
    latest_response_quality: ResponseQuality | None = None

    turn_count: int = Field(ge=0, default=0)
    dialogue: list[DialogueTurn] = Field(default_factory=list)
    history_summary: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # -- convenience accessors ------------------------------------------------
    def mastery_of(self, concept_id: str) -> float:
        entry = self.concept_mastery.get(concept_id)
        return entry.score if entry else 0.0

    def active_misconceptions(self) -> list[Misconception]:
        return [m for m in self.misconceptions if m.active]

    def find_misconception(self, concept_id: str) -> Misconception | None:
        for m in self.misconceptions:
            if m.concept_id == concept_id:
                return m
        return None

    def hints_for_current_exercise(self) -> list[HintRecord]:
        if not self.current_exercise_id:
            return list(self.previous_hints)
        return [h for h in self.previous_hints if h.exercise_id == self.current_exercise_id]

    def recent_dialogue(self, n: int) -> list[DialogueTurn]:
        return self.dialogue[-n:] if n > 0 else []

    def summary_for_prompt(self) -> dict[str, Any]:
        """Compact, LLM-facing projection of the state.

        Deliberately small: the model is conditioned on structured state, not on
        an ever-growing transcript.
        """
        return {
            "current_topic": self.current_topic,
            "current_exercise_id": self.current_exercise_id,
            "hint_level": self.hint_level,
            "attempt_count": self.attempt_count,
            "turn_count": self.turn_count,
            "concept_mastery": {
                cid: {
                    "score": round(cm.score, 2),
                    "confidence": round(cm.confidence, 2),
                    "evidence_count": cm.evidence_count,
                }
                for cid, cm in sorted(self.concept_mastery.items())
            },
            "active_misconceptions": [
                {
                    "concept_id": m.concept_id,
                    "description": m.description,
                    "confidence": round(m.confidence, 2),
                }
                for m in self.active_misconceptions()
            ],
            "demonstrated_concepts": self.demonstrated_concepts,
            "uncertain_concepts": self.uncertain_concepts,
            "previous_actions": [a.value for a in self.previous_actions[-6:]],
            "revealed_information_units": self.revealed_information_units,
            "latest_intent": self.latest_intent.value if self.latest_intent else None,
            "latest_response_quality": (
                self.latest_response_quality.value if self.latest_response_quality else None
            ),
            "history_summary": self.history_summary,
        }


# --------------------------------------------------------------------------- #
# Structured LLM outputs
# --------------------------------------------------------------------------- #


class MasteryEvidence(BaseModel):
    """Evidence about one concept observed in a single student message.

    The diagnosis LLM reports *evidence*, never an absolute mastery value. The
    deterministic updater owns the arithmetic, so the model cannot overwrite the
    learner model with an arbitrary number.
    """

    concept_id: str = Field(min_length=1)
    quality: ResponseQuality
    # How strongly this message speaks to the concept.
    weight: float = Field(ge=0.0, le=1.0, default=1.0)
    note: str = ""


class DetectedMisconception(BaseModel):
    concept_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)


class LearnerDiagnosis(BaseModel):
    """Structured output of the diagnosis stage."""

    intent: Intent
    response_quality: ResponseQuality = ResponseQuality.NOT_APPLICABLE
    mastery_evidence: list[MasteryEvidence] = Field(default_factory=list)
    detected_misconceptions: list[DetectedMisconception] = Field(default_factory=list)
    resolved_misconceptions: list[str] = Field(default_factory=list)
    current_topic: str | None = None
    current_exercise_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    @field_validator("mastery_evidence", "detected_misconceptions")
    @classmethod
    def _cap_list(cls, value: list[Any]) -> list[Any]:
        # A pathological model response cannot flood the learner model.
        return value[:8]


class PedagogicalDecision(BaseModel):
    """Structured output of the policy stage. Exactly one action."""

    action: PedagogicalAction
    target_concept: str | None = None
    needs_retrieval: bool = False
    retrieval_query: str | None = None
    desired_hint_level: int = Field(ge=0, default=0)
    reason_code: str = Field(default="unspecified", max_length=64)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    @model_validator(mode="after")
    def _retrieval_consistency(self) -> PedagogicalDecision:
        # RETRIEVE is meaningless without a query; normalise rather than crash so
        # a slightly sloppy model response is still usable.
        if self.action == PedagogicalAction.RETRIEVE:
            object.__setattr__(self, "needs_retrieval", True)
        if self.needs_retrieval and not (self.retrieval_query or "").strip():
            object.__setattr__(self, "retrieval_query", self.target_concept or "")
        if self.needs_retrieval and not (self.retrieval_query or "").strip():
            object.__setattr__(self, "needs_retrieval", False)
        return self


class RetrievedPassage(BaseModel):
    """A searchable-corpus passage. Never carries protected solution text."""

    chunk_id: str
    doc_id: str
    section_id: str | None = None
    section_title: str | None = None
    page: int | None = None
    text: str
    score: float = 0.0
    bm25_score: float = 0.0
    embedding_score: float = 0.0

    def citation(self) -> str:
        parts = [p for p in [self.section_id, self.section_title] if p]
        label = " ".join(parts) if parts else self.doc_id
        return f"{label} (p. {self.page})" if self.page else label


class ExerciseMeta(BaseModel):
    """Public exercise metadata. The official solution is never a field here."""

    exercise_id: str
    title: str
    question: str
    chapter: str | None = None
    concepts: list[str] = Field(default_factory=list)
    # Number of distinct answer units the official solution contains. A count is
    # safe to expose; the units themselves are not.
    protected_unit_count: int = 0


class DeterministicFinding(BaseModel):
    """Structured evidence from the deterministic leakage detector."""

    leaked: bool = False
    ngram_overlap: float = 0.0
    max_verbatim_run: int = 0
    covered_units: list[str] = Field(default_factory=list)
    unit_coverage: float = 0.0
    matched_phrases: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """Structured output of the independent semantic safety judge."""

    verdict: SafetyVerdict
    leaked_units: list[str] = Field(default_factory=list)
    reason_code: str = Field(default="unspecified", max_length=64)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class SafetyReport(BaseModel):
    """Combined result of the safety gate for one candidate response."""

    verdict: SafetyVerdict
    deterministic: DeterministicFinding = Field(default_factory=DeterministicFinding)
    judge: JudgeVerdict | None = None
    protected_exercise_active: bool = False
    reason_code: str = "no_protected_exercise"

    @property
    def leaked_units(self) -> list[str]:
        units = list(self.deterministic.covered_units)
        if self.judge:
            units.extend(u for u in self.judge.leaked_units if u not in units)
        return units


class TutorResponse(BaseModel):
    """What the runtime returns for one turn."""

    session_id: str
    turn_id: int
    text: str
    action: PedagogicalAction
    citations: list[str] = Field(default_factory=list)
    retrieved_section_ids: list[str] = Field(default_factory=list)
    safety_verdict: SafetyVerdict = SafetyVerdict.PASS
    revision_count: int = 0
    used_fallback: bool = False


class TurnMetrics(BaseModel):
    """Per-turn cost / efficiency observability."""

    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0


class SessionCreateResult(BaseModel):
    session_id: str
    created_at: datetime


ALL_ACTIONS: tuple[PedagogicalAction, ...] = tuple(PedagogicalAction)
