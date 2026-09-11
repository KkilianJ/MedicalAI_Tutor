"""Tool contract.

The LLM never executes code. It names a registered tool and supplies arguments,
the runtime validates those arguments against a Pydantic schema, executes the
tool itself, and converts the result into an `Observation`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from ..state.models import Observation


class ToolError(RuntimeError):
    """Raised when a tool cannot run or its arguments do not validate."""


class ToolResult(BaseModel):
    tool_name: str
    ok: bool = True
    payload: dict[str, Any] = {}
    error: str | None = None
    duration_ms: float = 0.0

    def to_observation(self, turn_id: int) -> Observation:
        return Observation.from_tool(
            self.tool_name,
            self.payload if self.ok else {"error": self.error},
            turn_id=turn_id,
            ok=self.ok,
        )

    def summary(self) -> dict[str, Any]:
        """Compact form stored in the trace (never the full passage text)."""
        if not self.ok:
            return {"ok": False, "error": self.error}
        summary: dict[str, Any] = {"ok": True}
        for key, value in self.payload.items():
            if isinstance(value, list):
                summary[key] = len(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value if not isinstance(value, str) else value[:120]
        return summary


class Tool(ABC):
    """Base class for every tool the agent may call."""

    name: str = "tool"
    description: str = ""
    args_schema: type[BaseModel]
    # Tools are opt-in per caller role; see ToolRegistry.
    permissions: frozenset[str] = frozenset({"tutor"})

    @abstractmethod
    def run(self, args: BaseModel) -> dict[str, Any]:
        """Execute with already-validated arguments."""

    def execute(self, raw_args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            args = self.args_schema.model_validate(raw_args or {})
        except ValidationError as exc:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=f"invalid arguments: {exc.error_count()} problem(s): {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        try:
            payload = self.run(args)
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        return ToolResult(
            tool_name=self.name,
            ok=True,
            payload=payload,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def spec(self) -> dict[str, Any]:
        """Description handed to the policy LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.args_schema.model_json_schema(),
        }
