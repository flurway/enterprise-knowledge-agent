"""
MCP adapter skeleton.

This module intentionally does not start a real MCP client yet. It gives MCP
tools a concrete dispatcher handler that fails with a structured unavailable
error, which is safer and more explainable than leaving registry entries with
no execution path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.tool_dispatcher import ToolDispatcher, ToolDispatchError, ToolInvocationContext
from agent.tool_registry import DEFAULT_TOOL_REGISTRY, ToolBackend, ToolRegistry, ToolSpec


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    enabled: bool = False
    command: str = ""
    url: str = ""
    metadata: dict[str, Any] | None = None


class McpToolUnavailableError(ToolDispatchError):
    pass


class McpToolAdapter:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        servers: dict[str, McpServerConfig] | None = None,
    ):
        self.registry = registry or DEFAULT_TOOL_REGISTRY
        self.servers = servers or {}

    def register_with(self, dispatcher: ToolDispatcher) -> list[str]:
        registered: list[str] = []
        for spec in self._mcp_tools():
            if dispatcher.has_handler(spec.name):
                continue
            dispatcher.register(spec.name, self._make_unavailable_handler(spec))
            registered.append(spec.name)
        return registered

    def manifest(self) -> dict[str, Any]:
        return {
            "servers": {
                name: {
                    "enabled": config.enabled,
                    "command": config.command,
                    "url": config.url,
                    "metadata": config.metadata or {},
                }
                for name, config in self.servers.items()
            },
            "tools": [
                {
                    "name": spec.name,
                    "mcp_server": spec.mcp_server,
                    "risk": spec.risk.value,
                    "requires_confirmation": spec.requires_confirmation,
                    "server_configured": spec.mcp_server in self.servers,
                    "server_enabled": self.servers.get(spec.mcp_server, McpServerConfig(spec.mcp_server)).enabled,
                }
                for spec in self._mcp_tools()
            ],
        }

    def _mcp_tools(self) -> list[ToolSpec]:
        return [spec for spec in self.registry.list_enabled() if spec.backend == ToolBackend.MCP]

    def _make_unavailable_handler(self, spec: ToolSpec):
        async def handler(context: ToolInvocationContext) -> Any:
            server = self.servers.get(spec.mcp_server)
            if server is None:
                raise McpToolUnavailableError(
                    f"MCP server '{spec.mcp_server}' is not configured for tool: {spec.name}",
                    tool_name=spec.name,
                    backend=ToolBackend.MCP.value,
                    code="mcp_server_unconfigured",
                )
            if not server.enabled:
                raise McpToolUnavailableError(
                    f"MCP server '{spec.mcp_server}' is configured but disabled for tool: {spec.name}",
                    tool_name=spec.name,
                    backend=ToolBackend.MCP.value,
                    code="mcp_server_disabled",
                )
            raise McpToolUnavailableError(
                f"MCP server '{spec.mcp_server}' has no client implementation for tool: {spec.name}",
                tool_name=spec.name,
                backend=ToolBackend.MCP.value,
                code="mcp_client_not_implemented",
            )

        return handler
