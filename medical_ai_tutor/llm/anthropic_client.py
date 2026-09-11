"""Anthropic provider.

Structured output uses provider-native tool calling: the model is given a single
tool whose input schema is the Pydantic JSON schema and is forced to call it.
That removes JSON-in-prose parsing from the control path entirely.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import get_secret
from .base import LLMClient, LLMError, LLMMessage, LLMResult

T = TypeVar("T", bound=BaseModel)

_EMIT_TOOL = "emit_structured_output"


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, timeout: float = 60.0) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise LLMError(
                "anthropic package not installed; run `pip install anthropic` or use the mock provider"
            ) from exc

        # get_secret(), not os.environ: a key written in local_settings.py must
        # work here too, not only when the factory passes it in explicitly.
        key = api_key or get_secret("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout)

    @staticmethod
    def _payload(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        return (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResult:
        try:
            response = self._client.messages.create(
                model=model,
                system=system,
                messages=self._payload(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"anthropic completion failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        input_tokens, output_tokens = self._usage(response)
        return LLMResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            raw=response,
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
        tool = {
            "name": _EMIT_TOOL,
            "description": f"Emit a valid {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }
        try:
            response = self._client.messages.create(
                model=model,
                system=system,
                messages=self._payload(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                tools=[tool],
                tool_choice={"type": "tool", "name": _EMIT_TOOL},
            )
        except Exception as exc:
            raise LLMError(f"anthropic structured call failed: {exc}") from exc

        input_tokens, output_tokens = self._usage(response)
        payload: dict[str, Any] | None = None
        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and block.name == _EMIT_TOOL:
                payload = dict(block.input)
                break
        if payload is None:
            raise LLMError("anthropic returned no tool_use block for structured output")

        result = LLMResult(
            text=str(payload),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            raw=response,
        )
        try:
            return schema.model_validate(payload), result
        except ValidationError as exc:
            # `except` names are unbound at block exit; capture before leaving.
            validation_error: Exception = exc
            if max_repair_attempts <= 0:
                raise LLMError(
                    f"anthropic structured output invalid: {validation_error}"
                ) from validation_error

        repair = list(messages) + [
            LLMMessage(
                role="user",
                content=(
                    f"The previous tool input failed validation: {validation_error}\n"
                    f"Call {_EMIT_TOOL} again with corrected values."
                ),
            )
        ]
        return self.structured(
            system, repair, schema, model, temperature, max_tokens, max_repair_attempts=0
        )
