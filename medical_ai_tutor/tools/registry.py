"""Tool registry with permission enforcement and per-turn call budgets."""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolResult

# Names that must never be registered as ordinary agent tools. The protected
# store is reachable only from the safety subsystem, by direct import, and this
# registry refuses to expose anything that looks like a route to it.
FORBIDDEN_TOOL_NAMES = frozenset(
    {"get_protected_solution", "get_solution", "read_protected", "official_solution"}
)


class ToolPermissionError(RuntimeError):
    """Raised when a caller requests a tool it is not permitted to use."""


class ToolBudgetExceeded(RuntimeError):
    """Raised when a turn exceeds its configured tool-call budget."""


class ToolRegistry:
    """Holds the tools the tutor may call and enforces how often."""

    def __init__(self, max_calls_per_turn: int = 3) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_calls_per_turn = max_calls_per_turn
        self._calls_this_turn = 0
        self.invalid_call_count = 0

    # -- registration ------------------------------------------------------ #
    def register(self, tool: Tool) -> ToolRegistry:
        if tool.name in FORBIDDEN_TOOL_NAMES:
            raise ToolPermissionError(
                f"tool {tool.name!r} may never be registered: protected solutions are "
                "accessible only to the safety subsystem"
            )
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [self._tools[name].spec() for name in self.names()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # -- execution --------------------------------------------------------- #
    def begin_turn(self) -> None:
        self._calls_this_turn = 0

    @property
    def calls_this_turn(self) -> int:
        return self._calls_this_turn

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_calls_per_turn - self._calls_this_turn)

    def execute(self, name: str, arguments: dict[str, Any], role: str = "tutor") -> ToolResult:
        """Validate permission and budget, then run the tool."""
        tool = self._tools.get(name)
        if tool is None:
            self.invalid_call_count += 1
            return ToolResult(
                tool_name=name,
                ok=False,
                error=f"unknown tool {name!r}; available: {self.names()}",
            )
        if role not in tool.permissions:
            self.invalid_call_count += 1
            return ToolResult(
                tool_name=name,
                ok=False,
                error=f"role {role!r} is not permitted to call {name!r}",
            )
        if self._calls_this_turn >= self.max_calls_per_turn:
            return ToolResult(
                tool_name=name,
                ok=False,
                error=(
                    f"tool budget exhausted: {self.max_calls_per_turn} calls per turn"
                ),
            )

        self._calls_this_turn += 1
        result = tool.execute(arguments)
        if not result.ok and "invalid arguments" in (result.error or ""):
            self.invalid_call_count += 1
        return result
