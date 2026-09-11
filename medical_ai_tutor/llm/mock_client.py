"""Deterministic mock provider.

The mock exists so the architecture, the tests, and the offline evaluation suite
run with zero API keys and zero network. It is not a toy stub: it implements
defensible heuristics for each LLM role so that pedagogical routing, learner
adaptation, and safety behaviour are all genuinely exercised offline.

Convention: every control-flow prompt passes its context as a JSON document in
the user message, so the mock can read the same structured context a real model
would see.
"""

from __future__ import annotations

import json
import re
import zlib
from typing import Any, TypeVar

from pydantic import BaseModel

from ..state.models import (
    DetectedMisconception,
    Intent,
    JudgeVerdict,
    LearnerDiagnosis,
    MasteryEvidence,
    PedagogicalAction,
    PedagogicalDecision,
    ResponseQuality,
    SafetyVerdict,
)
from .base import LLMClient, LLMMessage, LLMResult, extract_json

T = TypeVar("T", bound=BaseModel)

# --------------------------------------------------------------------------- #
# Lexicons driving the mock's judgments
# --------------------------------------------------------------------------- #

# Matched as a pattern rather than a fixed phrase list: a learner asking for
# "the official answer" or "the model solution" is making the same request as
# one asking to "just give me the answer", and a literal substring list misses
# every variant it was not written for.
_ANSWER_DEMAND_RE = re.compile(
    r"\b(?:give|show|tell|send|write|provide|hand|read)\s+(?:me\s+|us\s+)?"
    r"(?:the\s+|your\s+|its\s+)?"
    r"(?:official\s+|model\s+|sample\s+|full\s+|complete\s+|correct\s+|whole\s+|final\s+)*"
    r"(?:answers?|solutions?)\b"
    r"|\bwhat(?:'s| is| are)\s+the\s+(?:official\s+|correct\s+|right\s+)?(?:answers?|solutions?)\b"
    r"|\b(?:official|model|sample|complete|full)\s+(?:answers?|solutions?)\b"
    r"|\b(?:solve|answer|do)\s+(?:it|this|that|the exercise|the question)\s+for\s+me\b"
)
_INJECTION = (
    "ignore your",
    "ignore all previous",
    "ignore previous",
    "disregard your",
    "disregard previous",
    "system prompt",
    "you are now",
    "developer mode",
    "pretend you are",
    "act as if you have no",
    "override your",
    "reveal your instructions",
    "new instructions:",
)
_LEAK_PRESSURE = (
    "translate the solution",
    "as json",
    "as a table",
    "in bullet points",
    "paraphrase the solution",
    "role-play",
    "roleplay",
    "check my answer",
    "is this the full answer",
    "verify my complete answer",
)
_OFF_TOPIC = ("weather", "football", "recipe", "movie", "vacation", "joke", "stock price")
_ATTEMPT_MARKERS = (
    "i think",
    "my answer",
    "i would say",
    "is it because",
    "in my opinion",
    "i believe",
    "maybe",
    "i guess",
    "my understanding",
    "so it",
    "because",
)
_TEXTBOOK_MARKERS = (
    "textbook",
    "the book",
    "chapter",
    "according to",
    "definition of",
    "how does the book",
    "what does the book",
    "glossary",
    "3lgm",
    "figure",
    "cite",
)
# Domain words so common that their presence says nothing about leakage.
_UBIQUITOUS_TERMS = frozenset(
    """
    information system systems health healthcare data patient patients care medical hospital
    exercise question answer example course textbook chapter would could should about which
    where there their these those other another management model models process processes
    function functions component components
    """.split()
)

# Word-boundary tests: substring matching would fire "not" inside "another".
_NEGATION_RE = re.compile(r"\b(not|isn't|is not|aren't|cannot|can't|doesn't|don't|no longer)\b")
_JUSTIFICATION_RE = re.compile(r"\b(because|since|as it|so that|therefore|which means)\b")

# Flat conflation of concepts the course treats as distinct: a wrong answer
# rather than a partially right one.
_CONFLATION = (
    "is the same as",
    "are the same as",
    "are the same thing",
    "no difference between",
    "identical to",
    "means nothing more than",
    "is just another word for",
)

# "what is that mean?" is a clarification request, not a new concept question.
_CLARIFICATION_RE = re.compile(
    r"\bwhat (?:do|does|is|'s) (?:you|that|this|it) mean\b"
    r"|\bwhat do you mean\b"
    r"|\bcan you (?:clarify|rephrase|explain that|say that again)\b"
    r"|\bi (?:don't|do not|dont) (?:get|follow|understand) (?:that|this|it|you)\b"
    r"|\bcome again\b|\bhuh\b"
)

_CONFUSION = ("confused", "don't understand", "do not understand", "no idea", "lost", "unclear")

# Misconception patterns: (regex, concept_id, description)
_MISCONCEPTION_PATTERNS: list[tuple[str, str, str]] = [
    (
        r"normali[sz]ation\s+(just|only|simply)?\s*means?\s+splitting|splitting\s+a\s+large\s+table",
        "normalization",
        "Believes normalization is arbitrary table splitting rather than removal of "
        "redundancy driven by functional dependencies.",
    ),
    (
        r"(ehr|electronic health record)\s+is\s+(just|only|simply)\s+(a\s+)?(scan|pdf|digital copy)",
        "electronic_health_record",
        "Treats the electronic health record as a digitised paper document rather than "
        "structured, reusable data across the information system.",
    ),
    (
        r"interoperability\s+(just|only|simply)?\s*means?\s+(sending|transferring|exchanging)\s+(data|files|messages)",
        "interoperability",
        "Reduces interoperability to data transport, ignoring semantic and process "
        "interoperability.",
    ),
    (
        r"(one|a single)\s+(vendor|system|database)\s+(solves|fixes|removes)\s+(all|every)",
        "architectural_styles",
        "Assumes a monolithic single-vendor architecture eliminates all integration "
        "problems, ignoring its trade-offs.",
    ),
    (
        r"(hospital information system|his)\s+is\s+(just|only|simply)\s+(the\s+)?(software|it|computers)",
        "health_information_system",
        "Equates the health information system with its computer-based tools, omitting "
        "people, paper-based components and processes.",
    ),
]

_CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    "health_information_system": ("health information system", "his ", "hospital information system"),
    "electronic_health_record": ("electronic health record", "ehr", "patient record"),
    "interoperability": ("interoperability", "interoperable", "hl7", "fhir", "standard"),
    "integration": ("integration", "integrated", "communication server"),
    "architectural_styles": ("architecture", "architectural", "monolithic", "best of breed"),
    "data_information_knowledge": ("data, information", "information and knowledge", "knowledge"),
    "normalization": ("normalization", "normalisation", "redundancy", "functional dependency"),
    "information_management": ("information management", "strategic management", "cio", "governance"),
    "data_quality": ("data quality", "data integrity", "quality of data"),
    "3lgm2": ("3lgm", "three-layer", "domain layer", "logical tool layer", "physical tool layer"),
    "evaluation": ("evaluation", "study design", "questionnaire"),
    "data_protection": ("data protection", "privacy", "data security", "consent"),
}


def _lower(text: Any) -> str:
    return str(text or "").lower()


def _any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


class MockLLMClient(LLMClient):
    """Deterministic stand-in for a real provider.

    `script` allows an evaluation case or test to force specific structured
    outputs; entries are consumed in order and keyed by schema name.
    """

    name = "mock"

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script: list[dict[str, Any]] = list(script or [])
        self.calls: list[dict[str, Any]] = []
        # Set by tests to force a provider failure and exercise fallbacks.
        self.fail_on: set[str] = set()

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _context(messages: list[LLMMessage]) -> dict[str, Any]:
        for message in reversed(messages):
            if message.role != "user":
                continue
            try:
                return extract_json(message.content)
            except Exception:
                continue
        return {}

    def _take_script(self, key: str) -> dict[str, Any] | None:
        for index, entry in enumerate(self.script):
            if key in entry:
                return self.script.pop(index)[key]
        return None

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResult:
        context = self._context(messages)
        role = self._role_of(system)
        self.calls.append({"role": role, "model": model})
        if role in self.fail_on:
            raise RuntimeError(f"mock provider forced failure for role {role!r}")

        if role == "revision":
            text = self._revise(context)
        else:
            text = self._generate(context)
        return LLMResult(
            text=text,
            input_tokens=len(system) // 4 + sum(len(m.content) for m in messages) // 4,
            output_tokens=max(1, len(text) // 4),
            model=model,
        )

    def structured(
        self,
        system: str,
        messages: list[LLMMessage],
        schema: type[T],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_repair_attempts: int = 1,
    ) -> tuple[T, LLMResult]:
        context = self._context(messages)
        role = self._role_of(system)
        self.calls.append({"role": role, "model": model, "schema": schema.__name__})
        if role in self.fail_on:
            raise RuntimeError(f"mock provider forced failure for role {role!r}")

        scripted = self._take_script(schema.__name__)
        if scripted is not None:
            value = schema.model_validate(scripted)
        elif schema is LearnerDiagnosis:
            value = self._diagnose(context)  # type: ignore[assignment]
        elif schema is PedagogicalDecision:
            value = self._decide(context)  # type: ignore[assignment]
        elif schema is JudgeVerdict:
            value = self._judge(context)  # type: ignore[assignment]
        else:
            value = schema.model_validate({})  # type: ignore[assignment]

        payload = value.model_dump(mode="json")
        text = json.dumps(payload)
        return value, LLMResult(  # type: ignore[return-value]
            text=text,
            input_tokens=len(system) // 4 + sum(len(m.content) for m in messages) // 4,
            output_tokens=max(1, len(text) // 4),
            model=model,
        )

    @staticmethod
    def _role_of(system: str) -> str:
        marker = re.search(r"ROLE:\s*([a-z_]+)", system or "")
        return marker.group(1) if marker else "unknown"

    # ------------------------------------------------------------------ #
    # Role: diagnosis
    # ------------------------------------------------------------------ #
    def _diagnose(self, context: dict[str, Any]) -> LearnerDiagnosis:
        message = _lower(context.get("student_message"))
        state = context.get("learner_state") or {}
        exercise = context.get("exercise") or {}
        last_action = (state.get("previous_actions") or [None])[-1]

        intent = self._intent(message, last_action)
        concepts = self._concepts(message, state, exercise)
        misconceptions = self._misconceptions(message)

        quality = ResponseQuality.NOT_APPLICABLE
        if intent in (Intent.STUDENT_ATTEMPT, Intent.ANSWER_TO_TUTOR_QUESTION):
            if _any(message, _CONFLATION):
                quality = ResponseQuality.INCORRECT
            elif misconceptions:
                quality = ResponseQuality.PARTIALLY_CORRECT
            elif _any(message, _CONFUSION) or len(message.split()) < 4:
                quality = ResponseQuality.UNCLEAR
            elif _NEGATION_RE.search(message) and _JUSTIFICATION_RE.search(message):
                # A negation plus a justification is the shape of a corrected view.
                quality = ResponseQuality.CORRECT
            elif len(message.split()) > 22:
                quality = ResponseQuality.CORRECT
            else:
                quality = ResponseQuality.PARTIALLY_CORRECT

        evidence: list[MasteryEvidence] = []
        if quality != ResponseQuality.NOT_APPLICABLE:
            for concept in concepts[:3]:
                evidence.append(
                    MasteryEvidence(
                        concept_id=concept,
                        quality=quality,
                        weight=1.0 if concept == (concepts[0] if concepts else "") else 0.6,
                        note="mock heuristic evidence",
                    )
                )

        resolved: list[str] = []
        active = [m.get("concept_id") for m in state.get("active_misconceptions", [])]
        if quality == ResponseQuality.CORRECT:
            detected_ids = {m.concept_id for m in misconceptions}
            resolved = [cid for cid in active if cid and cid not in detected_ids]

        topic = concepts[0] if concepts else state.get("current_topic")
        return LearnerDiagnosis(
            intent=intent,
            response_quality=quality,
            mastery_evidence=evidence,
            detected_misconceptions=misconceptions,
            resolved_misconceptions=resolved,
            current_topic=topic,
            current_exercise_id=exercise.get("exercise_id") or state.get("current_exercise_id"),
            confidence=0.6,
        )

    @staticmethod
    def _intent(message: str, last_action: str | None) -> Intent:
        if _any(message, _INJECTION):
            return Intent.PROMPT_INJECTION
        if _ANSWER_DEMAND_RE.search(message) or _any(message, _LEAK_PRESSURE):
            return Intent.DIRECT_ANSWER_REQUEST
        if _any(message, _OFF_TOPIC):
            return Intent.OFF_TOPIC
        asked_question = last_action in {
            PedagogicalAction.ASK_SOCRATIC.value,
            PedagogicalAction.ASK_DIAGNOSTIC.value,
            PedagogicalAction.QUIZ.value,
            PedagogicalAction.CHALLENGE.value,
        }
        if asked_question and not message.strip().endswith("?"):
            return Intent.ANSWER_TO_TUTOR_QUESTION
        if _any(message, _ATTEMPT_MARKERS) and not message.strip().endswith("?"):
            return Intent.STUDENT_ATTEMPT
        if _CLARIFICATION_RE.search(message):
            return Intent.CLARIFICATION_REQUEST
        if "exercise" in message or re.search(r"\b\d+\.\d+\.\d+\b", message):
            return Intent.EXERCISE_HELP
        if message.strip().endswith("?") or message.strip().startswith(
            ("what", "why", "how", "when", "which", "who")
        ):
            return Intent.CONCEPT_QUESTION
        if message.strip():
            return Intent.STUDENT_ATTEMPT
        return Intent.UNKNOWN

    @staticmethod
    def _concepts(message: str, state: dict[str, Any], exercise: dict[str, Any]) -> list[str]:
        """Concepts the message touches, most salient first.

        Ranking by earliest mention rather than lexicon order matters: in "what
        is interoperability in a health information system?" both concepts
        match, but the subject of the question is the one the learner is asking
        about, and it is the one whose mastery should drive the decision.
        """
        matches: list[tuple[int, int, str]] = []
        for concept_id, needles in _CONCEPT_LEXICON.items():
            positions = [message.find(n) for n in needles if n in message]
            if positions:
                earliest = min(positions)
                longest = max(len(n) for n in needles if n in message)
                matches.append((earliest, -longest, concept_id))

        if matches:
            ranked = [concept_id for _, _, concept_id in sorted(matches)]
            # Continuity: if the session is already on one of these concepts,
            # keep working on it rather than switching topic mid-thread.
            topic = state.get("current_topic")
            if topic in ranked:
                ranked.remove(topic)
                ranked.insert(0, topic)
            return ranked

        for concept in exercise.get("concepts", []) or []:
            if concept:
                return [concept]
        topic = state.get("current_topic")
        return [topic] if topic else []

    @staticmethod
    def _misconceptions(message: str) -> list[DetectedMisconception]:
        out: list[DetectedMisconception] = []
        for pattern, concept_id, description in _MISCONCEPTION_PATTERNS:
            if re.search(pattern, message):
                out.append(
                    DetectedMisconception(
                        concept_id=concept_id, description=description, confidence=0.65
                    )
                )
        return out

    # ------------------------------------------------------------------ #
    # Role: pedagogical decision
    # ------------------------------------------------------------------ #
    def _decide(self, context: dict[str, Any]) -> PedagogicalDecision:
        state = context.get("learner_state") or {}
        diagnosis = context.get("diagnosis") or {}
        message = _lower(context.get("student_message"))
        exercise = context.get("exercise") or {}
        has_context = bool(context.get("has_retrieved_context"))

        intent = diagnosis.get("intent")
        quality = diagnosis.get("response_quality")
        topic = diagnosis.get("current_topic") or state.get("current_topic")
        mastery = state.get("concept_mastery") or {}
        score = float((mastery.get(topic) or {}).get("score", 0.0)) if topic else 0.0
        active = state.get("active_misconceptions") or []
        attempts = int(state.get("attempt_count", 0) or 0)
        hint_level = int(state.get("hint_level", 0) or 0)
        previous = state.get("previous_actions") or []
        exercise_active = bool(exercise.get("exercise_id"))
        textbook_specific = _any(message, _TEXTBOOK_MARKERS)

        def decision(
            action: PedagogicalAction,
            reason: str,
            *,
            retrieve: bool = False,
            query: str | None = None,
            hint: int = 0,
        ) -> PedagogicalDecision:
            return PedagogicalDecision(
                action=action,
                target_concept=topic,
                needs_retrieval=retrieve,
                retrieval_query=query,
                desired_hint_level=hint,
                reason_code=reason,
                confidence=0.7,
            )

        if intent == Intent.PROMPT_INJECTION.value:
            return decision(PedagogicalAction.REDIRECT, "injection_detected")
        if intent == Intent.OFF_TOPIC.value:
            return decision(PedagogicalAction.REDIRECT, "off_topic")
        if intent == Intent.DIRECT_ANSWER_REQUEST.value and exercise_active:
            return decision(PedagogicalAction.REFUSE_SOLUTION, "protected_solution_requested")

        # "Explain what you just said" comes before anything else: repeating the
        # previous action verbatim is exactly what the learner is complaining about.
        if intent == Intent.CLARIFICATION_REQUEST.value:
            return decision(PedagogicalAction.GIVE_EXAMPLE, "clarification_requested")

        # A live misconception outranks a general explanation, but only when it is
        # relevant to what the learner is doing *now*. An unscoped rule hijacks
        # every later turn: a misconception recorded about normalization would
        # answer a question about interoperability by talking about
        # normalization.
        message_concepts = set(self._concepts(message, state, exercise))
        relevant = [
            m
            for m in active
            if m.get("concept_id") == topic or m.get("concept_id") in message_concepts
        ]
        # Keep a correction thread alive: if the tutor just raised it and the
        # learner is replying, stay on it even though they named no concept.
        if (
            not relevant
            and previous[-1:] == [PedagogicalAction.CORRECT_MISCONCEPTION.value]
            and intent == Intent.ANSWER_TO_TUTOR_QUESTION.value
        ):
            relevant = active[:1]

        if relevant:
            first = relevant[0]
            return PedagogicalDecision(
                action=PedagogicalAction.CORRECT_MISCONCEPTION,
                target_concept=first.get("concept_id") or topic,
                needs_retrieval=False,
                desired_hint_level=hint_level,
                reason_code="active_misconception",
                confidence=0.75,
            )

        # Textbook-specific factual questions are the retrieval trigger.
        if textbook_specific and not has_context:
            return decision(
                PedagogicalAction.EXPLAIN_CONCEPT,
                "textbook_specific_question",
                retrieve=True,
                query=context.get("student_message") or topic or "",
            )

        if exercise_active and attempts == 0 and intent != Intent.CONCEPT_QUESTION.value:
            return decision(PedagogicalAction.ASK_DIAGNOSTIC, "no_attempt_yet")

        if quality == ResponseQuality.CORRECT.value and score >= 0.65:
            return decision(PedagogicalAction.CHALLENGE, "strong_understanding")
        if quality == ResponseQuality.CORRECT.value:
            return decision(PedagogicalAction.QUIZ, "consolidate_understanding")

        if quality == ResponseQuality.PARTIALLY_CORRECT.value:
            # Escalate only if a Socratic question has already been tried.
            if previous[-1:] == [PedagogicalAction.ASK_SOCRATIC.value] or hint_level > 0:
                return decision(
                    PedagogicalAction.GIVE_HINT, "partial_after_socratic", hint=hint_level + 1
                )
            return decision(PedagogicalAction.ASK_SOCRATIC, "partial_understanding")

        if quality == ResponseQuality.INCORRECT.value:
            return decision(PedagogicalAction.EXPLAIN_CONCEPT, "prerequisite_gap")
        if quality == ResponseQuality.UNCLEAR.value:
            return decision(PedagogicalAction.ASK_DIAGNOSTIC, "unclear_response")

        if intent == Intent.CONCEPT_QUESTION.value:
            if score >= 0.7:
                return decision(PedagogicalAction.CHALLENGE, "advanced_learner_question")
            if score >= 0.3:
                return decision(PedagogicalAction.ASK_SOCRATIC, "intermediate_learner_question")
            if has_context:
                return decision(PedagogicalAction.EXPLAIN_CONCEPT, "beginner_with_context")
            return decision(
                PedagogicalAction.EXPLAIN_CONCEPT,
                "beginner_needs_grounding",
                retrieve=True,
                query=context.get("student_message") or topic or "",
            )

        if intent == Intent.EXERCISE_HELP.value:
            return decision(PedagogicalAction.ASK_DIAGNOSTIC, "exercise_help_without_attempt")

        return decision(PedagogicalAction.ASK_DIAGNOSTIC, "default_probe")

    # ------------------------------------------------------------------ #
    # Role: safety judge
    # ------------------------------------------------------------------ #
    def _judge(self, context: dict[str, Any]) -> JudgeVerdict:
        det = context.get("deterministic_findings") or {}
        candidate = _lower(context.get("candidate_response"))
        units = context.get("protected_answer_units") or []
        covered = list(det.get("covered_units") or [])

        # Semantic pass: an answer unit counts as disclosed when its *distinctive*
        # terms show up in the candidate even though the wording differs.
        # Ubiquitous domain vocabulary is excluded: every legitimate tutor turn
        # says "health information system", and matching on that would flag
        # ordinary teaching as leakage.
        for unit in units:
            terms = [
                t
                for t in dict.fromkeys(re.findall(r"[a-z]{5,}", _lower(unit)))
                if t not in _UBIQUITOUS_TERMS
            ][:8]
            if len(terms) < 3:
                continue
            matches = sum(1 for t in terms if t in candidate)
            if matches >= max(3, int(len(terms) * 0.7)) and unit not in covered:
                covered.append(unit)

        if det.get("leaked") or len(covered) >= 2:
            verdict = SafetyVerdict.BLOCK if len(covered) >= 3 else SafetyVerdict.REVISE
            return JudgeVerdict(
                verdict=verdict,
                leaked_units=covered[:6],
                reason_code="answer_units_disclosed",
                confidence=0.8,
            )
        if covered:
            return JudgeVerdict(
                verdict=SafetyVerdict.REVISE,
                leaked_units=covered[:6],
                reason_code="partial_unit_disclosure",
                confidence=0.6,
            )
        return JudgeVerdict(verdict=SafetyVerdict.PASS, reason_code="no_leakage", confidence=0.7)

    # ------------------------------------------------------------------ #
    # Role: generation / revision
    # ------------------------------------------------------------------ #
    def _generate(self, context: dict[str, Any]) -> str:
        decision = context.get("decision") or {}
        action = decision.get("action", PedagogicalAction.ASK_DIAGNOSTIC.value)
        concept = (decision.get("target_concept") or context.get("current_topic") or "this topic")
        readable = str(concept).replace("_", " ")
        passages = context.get("passages") or []
        citation = passages[0].get("citation") if passages else None
        grounded = f" The textbook covers this in {citation}." if citation else ""

        # Pick a phrasing from the student's message, so two different questions
        # that route to the same action do not come back word-for-word identical.
        # Deterministic: the same message always yields the same variant.
        seed = zlib.crc32(_lower(context.get("student_message")).encode("utf-8"))

        alternates: dict[str, tuple[str, ...]] = {
            PedagogicalAction.ASK_SOCRATIC.value: (
                f"Suppose the situation you described happens twice in different parts of the "
                f"organisation. What would that imply for {readable}?",
                f"Take your answer one step further: what would break first if {readable} were "
                f"missing from a hospital that runs a dozen application components?",
                f"What would you have to observe in a real ward to be convinced that {readable} "
                f"is actually working?",
            ),
            PedagogicalAction.EXPLAIN_CONCEPT.value: (
                f"{readable.capitalize()} is best understood by asking which problem it solves in "
                f"a health information system.{grounded} The key point is the relationship "
                f"between the data recorded and the functions that need it.",
                f"Start from the problem: without {readable}, the same fact ends up recorded in "
                f"several places and nobody can say which is current.{grounded} That is what it "
                f"exists to prevent.",
                f"Think of {readable} as a property of the whole information system rather than "
                f"of one product.{grounded} It is about what the parts guarantee each other.",
            ),
            PedagogicalAction.CORRECT_MISCONCEPTION.value: (
                f"That is partly right, but it misses something important about {readable}. "
                f"The step you describe is a means, not the goal. What problem is the goal?",
                f"You have the mechanism right, but not yet what it is for. Describe a case where "
                f"you did exactly what you said and {readable} still was not achieved.",
                f"Half of that holds up. The part that does not: you are describing an action, "
                f"and {readable} is a property you have to end up with. What property?",
            ),
            PedagogicalAction.GIVE_EXAMPLE.value: (
                f"Concretely: a patient is admitted, and their address is recorded in two "
                f"application components. Use that case to reason about {readable}.",
                f"Picture a discharge letter that has to reach a GP's practice software. Walk "
                f"through what {readable} has to do for that to work.",
                f"Take a lab result travelling from the laboratory system to the ward. What does "
                f"{readable} have to guarantee along the way?",
            ),
        }

        options = alternates.get(action)
        if options:
            return options[seed % len(options)]

        templates = {
            PedagogicalAction.ASK_DIAGNOSTIC.value: (
                f"Before we go further, tell me how you would describe {readable} in your own "
                f"words, and where you get stuck."
            ),
            PedagogicalAction.ASK_SOCRATIC.value: (
                f"Suppose the situation you described happens twice in different parts of the "
                f"organisation. What would that imply for {readable}?"
            ),
            PedagogicalAction.GIVE_HINT.value: (
                f"Here is one cue: look at what {readable} is supposed to guarantee when the same "
                f"fact is stored in two places. Work from that single property."
            ),
            PedagogicalAction.GIVE_EXAMPLE.value: (
                f"Consider a concrete case: a patient is admitted, and their address is recorded "
                f"in two application components. Use that to reason about {readable}."
            ),
            PedagogicalAction.EXPLAIN_CONCEPT.value: (
                f"{readable.capitalize()} is best understood by asking which problem it solves in a "
                f"health information system.{grounded} The key point is the relationship between "
                f"the data recorded and the functions that need it."
            ),
            PedagogicalAction.CORRECT_MISCONCEPTION.value: (
                f"That is partly right, but it misses something important about {readable}. "
                f"The step you describe is a means, not the goal. What problem is the goal?"
            ),
            PedagogicalAction.CHALLENGE.value: (
                f"You have the core idea. Now apply it: how would {readable} change if the setting "
                f"were an ambulatory nursing organisation instead of a hospital?"
            ),
            PedagogicalAction.QUIZ.value: (
                f"Quick check: name two properties of {readable} and one situation where they "
                f"conflict."
            ),
            PedagogicalAction.RETRIEVE.value: (
                f"Let us ground this in the course material on {readable}.{grounded}"
            ),
            PedagogicalAction.REDIRECT.value: (
                "Let us stay with the course material. Which part of the health information "
                "systems topic would you like to work through?"
            ),
            PedagogicalAction.REFUSE_SOLUTION.value: (
                "I will not hand over the official solution, because working it out is the point. "
                f"Let us take the first step together: which aspect of {readable} does the "
                "exercise actually ask you to identify?"
            ),
        }
        return templates.get(action, templates[PedagogicalAction.ASK_DIAGNOSTIC.value])

    def _revise(self, context: dict[str, Any]) -> str:
        decision = context.get("decision") or {}
        concept = (decision.get("target_concept") or "this topic")
        readable = str(concept).replace("_", " ")
        return (
            "Let us approach this from your own reasoning instead. Take the first requirement the "
            f"exercise states about {readable}: which stakeholder does it serve, and what would go "
            "wrong if it were missing?"
        )
