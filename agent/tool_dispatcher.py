"""
Registry-backed tool dispatcher.

The dispatcher is the bridge between tool metadata and concrete handlers. It
keeps execution lookup separate from the executor's policy/hooks lifecycle, so
future MCP or subagent-backed tools can be wired in without expanding a long
if/elif chain in the hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agent.tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


@dataclass
class ToolInvocationContext:
    action: str
    params: dict[str, Any]
    step: dict[str, Any] = field(default_factory=dict)
    dep_context: str = ""
    stage_reminder: str = ""


ToolHandler = Callable[[ToolInvocationContext], Awaitable[Any]]


class ToolDispatchError(Exception):
    def __init__(self, message: str, tool_name: str = "", backend: str = "", code: str = "dispatch_error"):
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.backend = backend
        self.code = code

    def to_dict(self) -> dict[str, str]:
        return {
            "message": self.message,
            "tool_name": self.tool_name,
            "backend": self.backend,
            "code": self.code,
        }


class ToolDispatcher:
    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or DEFAULT_TOOL_REGISTRY
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        ok, reason = self.registry.validate_action(tool_name)
        if not ok:
            raise ToolDispatchError(
                f"Cannot register tool handler for {tool_name}: {reason}",
                tool_name=tool_name,
                code="tool_registration_rejected",
            )
        self._handlers[tool_name] = handler

    def has_handler(self, tool_name: str) -> bool:
        return tool_name in self._handlers

    def registered_tools(self) -> list[str]:
        return sorted(self._handlers)

    async def dispatch(self, action: str, context: ToolInvocationContext) -> Any:
        ok, reason = self.registry.validate_action(action)
        if not ok:
            raise ToolDispatchError(reason, tool_name=action, code="tool_validation_failed")

        handler = self._handlers.get(action)
        if handler is None:
            spec = self.registry.get(action)
            backend = spec.backend.value if spec else "unknown"
            raise ToolDispatchError(
                f"No handler registered for tool: {action} (backend={backend})",
                tool_name=action,
                backend=backend,
                code="tool_handler_missing",
            )

        return await handler(context)
