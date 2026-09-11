"""Provider-agnostic LLM interface.

Two call shapes exist:

* `complete()`   - free text, used only for the student-facing tutor message.
* `structured()` - schema-validated output, used for every control-flow
  decision (diagnosis, policy, safety judge).

Control flow never depends on parsing prose. If a provider returns something
that does not validate, exactly one schema-repair attempt is made and then the
call fails loudly so the caller can apply its own deterministic fallback.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a provider call fails or cannot produce valid output."""


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: Any = None


@dataclass
class UsageAccumulator:
    """Per-turn cost observability, incremented by the orchestrator."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_role: dict[str, int] = field(default_factory=dict)

    def add(self, result: LLMResult, role: str = "unknown") -> None:
        self.llm_calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.by_role[role] = self.by_role.get(role, 0) + 1


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Handles fenced blocks and leading prose, both of which providers emit even
    when told not to.
    """
    if not text or not text.strip():
        raise LLMError("empty model response")

    candidates: list[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())

    start = text.find("{")
    if start != -1:
        depth, in_string, escape = 0, False, False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMError(f"no JSON object found in response: {text[:200]!r}")


def schema_instructions(schema: type[BaseModel]) -> str:
    """Render the output contract appended to structured-output prompts."""
    return (
        "Respond with a single JSON object and nothing else. No prose, no code "
        "fence, no explanation. It must validate against this JSON schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


class LLMClient(ABC):
    """Base class for all providers."""

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResult: ...

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
        """Default structured output: instruct-and-parse with bounded repair.

        Providers with native structured output override this.
        """
        system_with_contract = f"{system}\n\n{schema_instructions(schema)}"
        result = self.complete(
            system_with_contract, messages, model, temperature=temperature, max_tokens=max_tokens
        )

        try:
            return schema.model_validate(extract_json(result.text)), result
        except (LLMError, ValidationError) as exc:
            # Python unbinds the `except` name at the end of the block, so the
            # failure detail has to be captured before leaving it.
            first_error: Exception = exc
            if max_repair_attempts <= 0:
                raise LLMError(f"structured output invalid: {first_error}") from first_error

        repair_messages = list(messages) + [
            LLMMessage(role="assistant", content=result.text),
            LLMMessage(
                role="user",
                content=(
                    "That response did not validate against the required schema:\n"
                    f"{first_error}\n\n"
                    "Return ONLY the corrected JSON object. No prose."
                ),
            ),
        ]
        repaired = self.complete(
            system_with_contract,
            repair_messages,
            model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        try:
            validated = schema.model_validate(extract_json(repaired.text))
        except (LLMError, ValidationError) as second_error:
            raise LLMError(f"structured output invalid after repair: {second_error}") from second_error

        # Report combined usage so repair attempts are visible in metrics.
        repaired.input_tokens += result.input_tokens
        repaired.output_tokens += result.output_tokens
        return validated, repaired
