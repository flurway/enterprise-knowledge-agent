"""
Tool registry and MCP-facing metadata.

The registry is the runtime's source of truth for tool capabilities. The
executor can still dispatch local Python functions, but planning, permission
checks, trace, and future MCP adapters should reason from this structured
metadata instead of scattered string allowlists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.policies import DEFAULT_TOOL_POLICIES, ToolRisk


class ToolScope(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    MEMORY = "memory"


class ToolBackend(str, Enum):
    LOCAL = "local"
    MCP = "mcp"
    SUBAGENT = "subagent"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    scopes: tuple[ToolScope, ...] = (ToolScope.READ,)
    risk: ToolRisk = ToolRisk.LOW
    backend: ToolBackend = ToolBackend.LOCAL
    enabled: bool = True
    requires_confirmation: bool = False
    mcp_server: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    def is_read_only(self) -> bool:
        write_scopes = {ToolScope.WRITE, ToolScope.EXECUTE, ToolScope.MEMORY}
        return not any(scope in write_scopes for scope in self.scopes)

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "backend": self.backend.value,
            "enabled": self.enabled,
            "risk": self.risk.value,
            "scopes": [scope.value for scope in self.scopes],
            "requires_confirmation": self.requires_confirmation,
            "mcp_server": self.mcp_server,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None):
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def has_enabled(self, name: str) -> bool:
        spec = self.get(name)
        return bool(spec and spec.enabled)

    def list_enabled(self) -> list[ToolSpec]:
        return [spec for spec in self._tools.values() if spec.enabled]

    def list_by_scope(self, scope: ToolScope) -> list[ToolSpec]:
        return [spec for spec in self.list_enabled() if scope in spec.scopes]

    def validate_action(self, action: str) -> tuple[bool, str]:
        spec = self.get(action)
        if spec is None:
            return False, f"unknown tool action: {action}"
        if not spec.enabled:
            return False, f"tool action is disabled: {action}"
        return True, "tool action registered"

    def to_mcp_manifest(self) -> dict[str, Any]:
        return {
            "version": "0.1",
            "tools": [spec.to_manifest_entry() for spec in self.list_enabled()],
        }


DEFAULT_TOOL_SPECS = [
    ToolSpec(
        name="search_knowledge_base",
        description="Read-only local knowledge retrieval",
        scopes=(ToolScope.READ,),
        risk=ToolRisk.LOW,
        input_schema={"query": "string", "top_k": "integer"},
    ),
    ToolSpec(
        name="read_document_detail",
        description="Read-only local document detail lookup",
        scopes=(ToolScope.READ,),
        risk=ToolRisk.LOW,
        input_schema={"doc_id": "string", "section": "string"},
    ),
    ToolSpec(
        name="summarize_content",
        description="LLM summarization over provided material",
        scopes=(ToolScope.READ,),
        risk=ToolRisk.LOW,
    ),
    ToolSpec(
        name="compare_concepts",
        description="LLM comparison over provided material",
        scopes=(ToolScope.READ,),
        risk=ToolRisk.LOW,
    ),
    ToolSpec(
        name="generate_research_report",
        description="Final synthesis over gathered material",
        scopes=(ToolScope.READ,),
        risk=ToolRisk.LOW,
    ),
    ToolSpec(
        name="ask_user_clarification",
        description="Ask the user for missing information",
        scopes=(ToolScope.READ,),
        risk=ToolRisk.LOW,
    ),
    ToolSpec(
        name="web_search",
        description="External search request",
        scopes=(ToolScope.READ, ToolScope.NETWORK),
        risk=ToolRisk.MEDIUM,
        input_schema={"query": "string", "max_results": "integer", "search_type": "string"},
    ),
    ToolSpec(
        name="fetch_webpage",
        description="External webpage fetch",
        scopes=(ToolScope.READ, ToolScope.NETWORK),
        risk=ToolRisk.MEDIUM,
        input_schema={"url": "string"},
    ),
    ToolSpec(
        name="memory_write",
        description="Persistent memory write",
        scopes=(ToolScope.MEMORY, ToolScope.WRITE),
        risk=ToolRisk.HIGH,
        requires_confirmation=True,
    ),
    ToolSpec(
        name="code_search",
        description="Read-only codebase search tool exposed as a future MCP adapter",
        scopes=(ToolScope.READ,),
        risk=ToolRisk.LOW,
        backend=ToolBackend.MCP,
        mcp_server="code",
        input_schema={"query": "string", "path": "string"},
    ),
    ToolSpec(
        name="run_tests",
        description="Run allowlisted project test commands",
        scopes=(ToolScope.EXECUTE,),
        risk=ToolRisk.MEDIUM,
        backend=ToolBackend.MCP,
        mcp_server="code",
        requires_confirmation=True,
        input_schema={"command": "string"},
    ),
    ToolSpec(
        name="code_patch",
        description="Modify source files through a constrained patch tool",
        scopes=(ToolScope.WRITE,),
        risk=ToolRisk.HIGH,
        backend=ToolBackend.MCP,
        mcp_server="code",
        requires_confirmation=True,
        input_schema={"files": "array", "patch": "string"},
    ),
]


DEFAULT_TOOL_REGISTRY = ToolRegistry(DEFAULT_TOOL_SPECS)


def policy_registry_mismatches() -> list[str]:
    """Return tool names that are present in policy but missing registry, or vice versa."""
    policy_names = set(DEFAULT_TOOL_POLICIES)
    registry_names = {spec.name for spec in DEFAULT_TOOL_SPECS}
    missing_from_registry = sorted(policy_names - registry_names)
    missing_from_policy = sorted(registry_names - policy_names)
    return [
        *(f"policy_without_registry:{name}" for name in missing_from_registry),
        *(f"registry_without_policy:{name}" for name in missing_from_policy),
    ]
