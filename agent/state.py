"""
Agent runtime state and trace models.

These models make a run inspectable without forcing the whole orchestrator
to move to LangGraph immediately. They are deliberately plain dataclasses so
they can be serialized to JSON, stored in SQLite/MySQL later, or mapped onto a
LangGraph state object when the project is ready for that migration.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class FailureType(str, Enum):
    NONE = "none"
    TOOL_ERROR = "tool_error"
    EMPTY_RESULT = "empty_result"
    POLICY_BLOCKED = "policy_blocked"
    TIMEOUT = "timeout"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


@dataclass
class TraceEvent:
    event_type: str
    message: str
    timestamp_ms: int = field(default_factory=now_ms)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    tool_name: str
    step_id: int
    input_preview: str = ""
    output_preview: str = ""
    success: bool = False
    error: str = ""
    latency_ms: int = 0
    risk_level: str = "low"
    policy_decision: str = "not_checked"
    timestamp_ms: int = field(default_factory=now_ms)


@dataclass
class PlanStepState:
    step_id: int
    action: str
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    depends_on: list[int] = field(default_factory=list)
    expected_output: str = ""
    verification: str = ""
    failure_type: FailureType = FailureType.NONE
    error: str = ""
    started_at_ms: Optional[int] = None
    finished_at_ms: Optional[int] = None

    @classmethod
    def from_plan_step(cls, step: dict[str, Any]) -> "PlanStepState":
        return cls(
            step_id=int(step.get("step_id", 0) or 0),
            action=str(step.get("action", "")),
            description=str(step.get("description", "")),
            depends_on=list(step.get("depends_on", []) or []),
            expected_output=str(step.get("expected_output", "")),
            verification=str(step.get("verification", "")),
        )


@dataclass
class AgentRun:
    session_id: str
    user_message: str
    run_id: str = field(default_factory=new_run_id)
    intent: str = ""
    refined_query: str = ""
    status: StepStatus = StepStatus.PENDING
    plan: dict[str, Any] = field(default_factory=dict)
    steps: list[PlanStepState] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    started_at_ms: int = field(default_factory=now_ms)
    finished_at_ms: Optional[int] = None

    def add_event(self, event_type: str, message: str, **metadata: Any) -> None:
        self.events.append(TraceEvent(event_type=event_type, message=message, metadata=metadata))

    def set_plan(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.steps = [PlanStepState.from_plan_step(step) for step in plan.get("steps", [])]
        self.add_event("plan_created", f"Plan contains {len(self.steps)} step(s)")

    def record_tool_call(self, call: ToolCallRecord) -> None:
        self.tool_calls.append(call)

    def finish(self, success: bool) -> None:
        self.status = StepStatus.SUCCEEDED if success else StepStatus.FAILED
        self.finished_at_ms = now_ms()
        self.add_event("run_finished", "Agent run finished", success=success)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        for step in data["steps"]:
            if isinstance(step.get("status"), StepStatus):
                step["status"] = step["status"].value
            if isinstance(step.get("failure_type"), FailureType):
                step["failure_type"] = step["failure_type"].value
        return data

