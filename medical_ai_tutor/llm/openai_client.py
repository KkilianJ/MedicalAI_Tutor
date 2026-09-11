"""OpenAI-compatible provider.

Structured output uses JSON mode plus the base class's instruct-and-validate
path, which keeps the client usable against non-OpenAI endpoints that implement
the same API surface (vLLM, Together, Azure, local gateways).
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import get_secret
from .base import LLMClient, LLMError, LLMMessage, LLMResult, extract_json, schema_instructions

T = TypeVar("T", bound=BaseModel)


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "openai package not installed; run `pip install openai` or use the mock provider"
            ) from exc

        key = api_key or get_secret("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = openai.OpenAI(
            api_key=key,
            base_url=base_url or get_secret("OPENAI_BASE_URL") or None,
            timeout=timeout,
        )

    @staticmethod
    def _payload(system: str, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        return [{"role": "system", "content": system}] + [
            {"role": m.role, "content": m.content} for m in messages
        ]

    def _call(
        self,
        system: str,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._payload(system, messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"openai call failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        return LLMResult(
            text=response.choices[0].message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=model,
            raw=response,
        )

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResult:
        return self._call(system, messages, model, temperature, max_tokens, json_mode=False)

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
        system_with_contract = f"{system}\n\n{schema_instructions(schema)}"
        result = self._call(
            system_with_contract, messages, model, temperature, max_tokens, json_mode=True
        )
        try:
            return schema.model_validate(extract_json(result.text)), result
        except Exception as exc:
            # `except` names are unbound at block exit; capture before leaving.
            first_error: Exception = exc
            if max_repair_attempts <= 0:
                raise LLMError(f"openai structured output invalid: {first_error}") from first_error

        repair = list(messages) + [
            LLMMessage(role="assistant", content=result.text),
            LLMMessage(
                role="user",
                content=f"That JSON failed validation: {first_error}\nReturn only corrected JSON.",
            ),
        ]
        repaired = self._call(
            system_with_contract, repair, model, 0.0, max_tokens, json_mode=True
        )
        try:
            validated = schema.model_validate(extract_json(repaired.text))
        except Exception as exc:
            raise LLMError(f"openai structured output invalid after repair: {exc}") from exc
        repaired.input_tokens += result.input_tokens
        repaired.output_tokens += result.output_tokens
        return validated, repaired
