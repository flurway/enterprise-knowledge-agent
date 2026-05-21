"""
Tool execution policy.

The current project does not yet execute dangerous local code, but adding this
policy layer early keeps later MCP/code-agent integration from becoming a
permission free-for-all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ToolPolicySpec:
    risk: ToolRisk
    description: str
    allow_from_untrusted_context: bool = False
    requires_confirmation: bool = False


@dataclass
class ToolPolicyDecision:
    allowed: bool
    reason: str
    risk: ToolRisk = ToolRisk.LOW

    @property
    def status(self) -> str:
        return "allowed" if self.allowed else "blocked"

    def to_dict(self) -> dict[str, str | bool]:
        return {"allowed": self.allowed, "reason": self.reason, "risk": self.risk.value}


DEFAULT_TOOL_POLICIES: dict[str, ToolPolicySpec] = {
    "search_knowledge_base": ToolPolicySpec(ToolRisk.LOW, "Read-only local knowledge retrieval"),
    "read_document_detail": ToolPolicySpec(ToolRisk.LOW, "Read-only local document detail lookup"),
    "summarize_content": ToolPolicySpec(ToolRisk.LOW, "LLM summarization over provided material"),
    "compare_concepts": ToolPolicySpec(ToolRisk.LOW, "LLM comparison over provided material"),
    "generate_research_report": ToolPolicySpec(ToolRisk.LOW, "Final synthesis over gathered material"),
    "ask_user_clarification": ToolPolicySpec(ToolRisk.LOW, "Ask the user for missing information"),
    "web_search": ToolPolicySpec(ToolRisk.MEDIUM, "External search request"),
    "fetch_webpage": ToolPolicySpec(ToolRisk.MEDIUM, "External webpage fetch"),
    "memory_write": ToolPolicySpec(ToolRisk.HIGH, "Persistent memory write", requires_confirmation=True),
    "code_search": ToolPolicySpec(ToolRisk.LOW, "Read-only codebase search"),
    "code_patch": ToolPolicySpec(ToolRisk.HIGH, "Modify source files", requires_confirmation=True),
    "run_tests": ToolPolicySpec(ToolRisk.MEDIUM, "Run project test commands", requires_confirmation=True),
}


class ToolPolicy:
    def __init__(
        self,
        policies: dict[str, ToolPolicySpec] | None = None,
        allow_high_risk: bool = False,
    ):
        self.policies = policies or DEFAULT_TOOL_POLICIES
        self.allow_high_risk = allow_high_risk

    def check(
        self,
        action: str,
        params: dict[str, Any] | None = None,
        triggered_by_untrusted_context: bool = False,
    ) -> ToolPolicyDecision:
        params = params or {}
        spec = self.policies.get(action)
        if spec is None:
            return ToolPolicyDecision(False, f"Unknown tool action: {action}", ToolRisk.HIGH)

        if triggered_by_untrusted_context and not spec.allow_from_untrusted_context:
            return ToolPolicyDecision(
                False,
                "Tool call was triggered by untrusted context and this tool is not allowlisted for that source",
                spec.risk,
            )

        if spec.requires_confirmation and not self.allow_high_risk:
            return ToolPolicyDecision(
                False,
                "Tool requires explicit confirmation or a dedicated execution mode",
                spec.risk,
            )

        if action == "fetch_webpage":
            from security.url_safety import check_url_safety

            decision = check_url_safety(str(params.get("url", "")))
            if not decision.allowed:
                return ToolPolicyDecision(False, decision.reason, spec.risk)

        return ToolPolicyDecision(True, "Tool call allowed by policy", spec.risk)


def preview_payload(payload: Any, max_chars: int = 500) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(payload)
    return text[:max_chars]
