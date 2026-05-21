"""
Explicit Agent runtime helpers.

This module is the first Phase 2 step: it makes orchestration state explicit
without replacing the existing orchestrator in one risky rewrite. The runtime
owns node transitions, plan validation, dependency checks, and step status
updates on AgentRun.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from agent.state import AgentRun, FailureType, PlanStepState, StepStatus, now_ms
from agent.tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry
from agent.stage_reminder import StageReminderManager, default_stage_constraints


@dataclass
class RuntimeStepResult:
    step_id: int
    action: str
    success: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "success": self.success,
            "output": self.output[:500],
            "error": self.error,
        }


class RuntimeNode(str, Enum):
    CLASSIFY = "classify"
    RETRIEVE_MEMORY = "retrieve_memory"
    DIRECT_SEARCH = "direct_search"
    PLAN = "plan"
    VALIDATE_PLAN = "validate_plan"
    EXECUTE_STEP = "execute_step"
    REPLAN = "replan"
    REFLECT = "reflect"
    SYNTHESIZE = "synthesize"
    VALIDATE_ANSWER = "validate_answer"
    SAVE_MEMORY = "save_memory"
    FINISH = "finish"


class RuntimeMode(str, Enum):
    EXPLORE = "explore"
    PLAN = "plan"
    EXECUTE = "execute"
    REVIEW = "review"


class PermissionMode(str, Enum):
    PLAN_ONLY = "plan_only"
    READ_ONLY = "read_only"
    ASK = "ask"
    AUTO = "auto"


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class TodoItem:
    todo_id: str
    content: str
    status: TodoStatus = TodoStatus.PENDING
    source: str = "plan"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "todo_id": self.todo_id,
            "content": self.content,
            "status": self.status.value,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class ReActObservation:
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActDecision:
    action: str
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    todo_id: str = ""
    reason: str = ""
    done: bool = False


class AgentRuntime:
    def __init__(self, run: AgentRun, tool_registry: ToolRegistry | None = None,
                 stage_reminder: StageReminderManager | None = None):
        self.run = run
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY
        self.current_node: RuntimeNode | None = None
        self.mode = RuntimeMode.EXPLORE
        self.permission_mode = PermissionMode.ASK
        self.todos: list[TodoItem] = []
        self.observations: list[ReActObservation] = []
        self.stage_reminder = stage_reminder or StageReminderManager()

    def enter_node(self, node: RuntimeNode, message: str | None = None, **metadata: Any) -> None:
        self.current_node = node
        self.run.add_event(
            "node_entered",
            message or f"Entered runtime node: {node.value}",
            node=node.value,
            **metadata,
        )

    def set_mode(self, mode: RuntimeMode, reason: str = "") -> None:
        previous = self.mode
        self.mode = mode
        self.run.add_event(
            "mode_changed",
            f"Runtime mode changed from {previous.value} to {mode.value}",
            previous=previous.value,
            current=mode.value,
            reason=reason,
        )

    def set_permission_mode(self, mode: PermissionMode, reason: str = "") -> None:
        previous = self.permission_mode
        self.permission_mode = mode
        self.run.add_event(
            "permission_mode_changed",
            f"Permission mode changed from {previous.value} to {mode.value}",
            previous=previous.value,
            current=mode.value,
            reason=reason,
        )

    def add_todo(self, content: str, source: str = "plan", **metadata: Any) -> TodoItem:
        todo = TodoItem(
            todo_id=f"todo-{len(self.todos) + 1}",
            content=content,
            source=source,
            metadata=metadata,
        )
        self.todos.append(todo)
        self.run.add_event("todo_added", content, todo=todo.to_dict())
        return todo

    def load_plan_as_todos(self, plan: dict[str, Any]) -> None:
        for step in plan.get("steps", []) or []:
            description = step.get("description") or f"{step.get('action', '')}: {step.get('expected_output', '')}"
            self.add_todo(
                content=description,
                source="plan",
                step_id=step.get("step_id"),
                action=step.get("action"),
            )

    def todo_for_step(self, step_id: int) -> TodoItem | None:
        for todo in self.todos:
            if todo.metadata.get("step_id") == step_id:
                return todo
        return None

    def update_todo(self, todo_id: str, status: TodoStatus, reason: str = "") -> None:
        todo = self.find_todo(todo_id)
        if todo is None:
            return
        previous = todo.status
        todo.status = status
        self.run.add_event(
            "todo_updated",
            f"Todo {todo_id} changed from {previous.value} to {status.value}",
            todo=todo.to_dict(),
            previous=previous.value,
            reason=reason,
        )

    def find_todo(self, todo_id: str) -> TodoItem | None:
        for todo in self.todos:
            if todo.todo_id == todo_id:
                return todo
        return None

    def next_pending_todo(self) -> TodoItem | None:
        for todo in self.todos:
            if todo.status == TodoStatus.PENDING:
                return todo
        return None

    def observe(self, summary: str, **metadata: Any) -> ReActObservation:
        observation = ReActObservation(summary=summary, metadata=metadata)
        self.observations.append(observation)
        self.run.add_event("react_observed", summary, metadata=metadata)
        return observation

    def decide_next(self) -> ReActDecision:
        todo = self.next_pending_todo()
        if todo is None:
            decision = ReActDecision(action="finish", reason="No pending todos", done=True)
        elif self.mode == RuntimeMode.PLAN:
            decision = ReActDecision(
                action="finish",
                todo_id=todo.todo_id,
                reason="Plan mode stops before tool execution",
                done=True,
            )
        else:
            decision = ReActDecision(
                action="work_on_todo",
                todo_id=todo.todo_id,
                reason="Next pending todo selected",
            )
        self.run.add_event("react_decided", decision.reason, decision=decision.__dict__)
        return decision

    def check_permission_for_tool(self, tool_name: str) -> tuple[bool, str]:
        spec = self.tool_registry.get(tool_name)
        if spec is None:
            return False, f"Unknown tool action: {tool_name}"
        if not spec.enabled:
            return False, f"Tool action is disabled: {tool_name}"
        if self.permission_mode == PermissionMode.PLAN_ONLY:
            return False, "Plan-only mode blocks tool execution"
        if self.permission_mode == PermissionMode.READ_ONLY and not spec.is_read_only():
            return False, f"Read-only mode blocks tool: {tool_name}"
        if self.permission_mode == PermissionMode.ASK:
            return False, "Ask mode requires explicit user approval before tool execution"
        if spec.requires_confirmation and self.permission_mode != PermissionMode.AUTO:
            return False, f"Tool requires confirmation: {tool_name}"
        return True, "Permission mode allows tool execution"

    def set_plan(self, plan: dict[str, Any]) -> None:
        self.run.set_plan(plan)

    def validate_plan(self, plan: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            errors.append("plan.steps must be a list")
            self._record_plan_validation(errors)
            return errors

        seen: set[int] = set()
        for raw_step in steps:
            step_id = int(raw_step.get("step_id", 0) or 0)
            action = str(raw_step.get("action", ""))
            if step_id <= 0:
                errors.append(f"step has invalid step_id: {step_id}")
            if step_id in seen:
                errors.append(f"duplicate step_id: {step_id}")
            seen.add(step_id)
            action_ok, reason = self.tool_registry.validate_action(action)
            if not action_ok:
                label = "unknown action" if "unknown" in reason else "invalid action"
                errors.append(f"step {step_id} uses {label}: {action} ({reason})")

        for raw_step in steps:
            step_id = int(raw_step.get("step_id", 0) or 0)
            for dep_id in raw_step.get("depends_on", []) or []:
                if dep_id not in seen:
                    errors.append(f"step {step_id} depends on missing step {dep_id}")
                elif dep_id == step_id:
                    errors.append(f"step {step_id} cannot depend on itself")

        self._record_plan_validation(errors)
        return errors

    def _record_plan_validation(self, errors: list[str]) -> None:
        self.run.add_event(
            "plan_validated",
            "Plan validation completed",
            valid=not errors,
            errors=errors,
        )

    def can_execute_step(self, step_id: int) -> tuple[bool, list[int]]:
        step = self._find_step(step_id)
        if step is None:
            return False, []
        unmet = [
            dep_id for dep_id in step.depends_on
            if not self._is_step_succeeded(dep_id)
        ]
        return not unmet, unmet

    def mark_step_running(self, step_id: int) -> None:
        step = self._find_step(step_id)
        if step is None:
            return
        step.status = StepStatus.RUNNING
        step.started_at_ms = now_ms()
        self.run.add_event("step_started", f"Step {step_id} started", step_id=step_id, action=step.action)

    def mark_step_finished(self, step_id: int, success: bool, error: str = "") -> None:
        step = self._find_step(step_id)
        if step is None:
            return
        step.status = StepStatus.SUCCEEDED if success else StepStatus.FAILED
        step.failure_type = FailureType.NONE if success else self._classify_failure(error)
        step.error = error
        step.finished_at_ms = now_ms()
        self.run.add_event(
            "step_finished",
            f"Step {step_id} {'succeeded' if success else 'failed'}",
            step_id=step_id,
            action=step.action,
            success=success,
            failure_type=step.failure_type.value,
            error=error,
        )

    def mark_step_skipped(self, step_id: int, reason: str, unmet_dependencies: list[int] | None = None) -> None:
        step = self._find_step(step_id)
        if step is None:
            return
        step.status = StepStatus.SKIPPED
        step.failure_type = FailureType.UNKNOWN
        step.error = reason
        step.finished_at_ms = now_ms()
        self.run.add_event(
            "step_skipped",
            reason,
            step_id=step_id,
            action=step.action,
            unmet_dependencies=unmet_dependencies or [],
        )

    def _find_step(self, step_id: int) -> PlanStepState | None:
        for step in self.run.steps:
            if step.step_id == step_id:
                return step
        return None

    def _is_step_succeeded(self, step_id: int) -> bool:
        step = self._find_step(step_id)
        return bool(step and step.status == StepStatus.SUCCEEDED)

    @staticmethod
    def _classify_failure(error: str) -> FailureType:
        lowered = error.lower()
        if "policy blocked" in lowered:
            return FailureType.POLICY_BLOCKED
        if "no results" in lowered or "未找到" in error:
            return FailureType.EMPTY_RESULT
        if "timeout" in lowered or "timed out" in lowered:
            return FailureType.TIMEOUT
        if error:
            return FailureType.TOOL_ERROR
        return FailureType.UNKNOWN


class ReActLoop:
    """
    Todo-aware ReAct loop skeleton.

    It is intentionally small: the LLM is not yet choosing arbitrary tools.
    Instead, the loop observes pending todos, maps each todo back to a planned
    step, checks runtime permission, executes the step callback, and updates
    todo/step state. This is the migration bridge from Plan-and-Execute to a
    full ReAct runtime.
    """

    def __init__(self, runtime: AgentRuntime, max_iterations: int = 20):
        self.runtime = runtime
        self.max_iterations = max_iterations

    async def run_plan_todos(
        self,
        steps_by_id: dict[int, dict[str, Any]],
        execute_step: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> list[Any]:
        results: list[Any] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            self.runtime.observe(
                "ReAct loop iteration",
                iteration=iterations,
                pending_todos=len([todo for todo in self.runtime.todos if todo.status == TodoStatus.PENDING]),
            )
            decision = self.runtime.decide_next()
            if decision.done:
                break

            todo = self.runtime.find_todo(decision.todo_id)
            if todo is None:
                break

            step_id = int(todo.metadata.get("step_id", 0) or 0)
            step = steps_by_id.get(step_id)
            if not step:
                self.runtime.update_todo(todo.todo_id, TodoStatus.BLOCKED, "Todo is not linked to a plan step")
                results.append(RuntimeStepResult(step_id, str(todo.metadata.get("action", "")), False, error="Todo is not linked to a plan step"))
                continue

            action = str(step.get("action", ""))
            allowed, reason = self.runtime.check_permission_for_tool(action)
            if not allowed:
                self.runtime.update_todo(todo.todo_id, TodoStatus.BLOCKED, reason)
                self.runtime.mark_step_skipped(step_id, reason)
                results.append(RuntimeStepResult(step_id, action, False, error=reason))
                continue

            can_execute, unmet = self.runtime.can_execute_step(step_id)
            if not can_execute:
                reason = f"Step {step_id} skipped because dependencies are not satisfied: {unmet}"
                self.runtime.update_todo(todo.todo_id, TodoStatus.BLOCKED, reason)
                self.runtime.mark_step_skipped(step_id, reason, unmet_dependencies=unmet)
                results.append(RuntimeStepResult(step_id, action, False, error=reason))
                continue

            self.runtime.update_todo(todo.todo_id, TodoStatus.IN_PROGRESS, "Executing linked plan step")
            self.runtime.enter_node(RuntimeNode.EXECUTE_STEP, "Executing todo-linked plan step", step_id=step_id, action=action, todo_id=todo.todo_id)
            self.runtime.mark_step_running(step_id)

            # Stage-aware constraint injection: combat attention decay by
            # re-injecting phase-specific guardrails into recent context.
            reminder = self.runtime.stage_reminder.build_reminder(step)
            if reminder:
                step["_stage_reminder"] = reminder

            result = await execute_step(step)
            self.runtime.mark_step_finished(step_id, result.success, result.error)

            # Feed result back for rush detection
            output = getattr(result, "output", "") or ""
            data = getattr(result, "data", None) or {}
            citations = data.get("citations", {}) if isinstance(data, dict) else {}
            self.runtime.stage_reminder.record_step_result(output, len(citations) if isinstance(citations, dict) else 0)

            self.runtime.update_todo(
                todo.todo_id,
                TodoStatus.DONE if result.success else TodoStatus.BLOCKED,
                "Step succeeded" if result.success else result.error,
            )
            results.append(result)

            if not result.success:
                break

        if iterations >= self.max_iterations:
            self.runtime.run.add_event("react_loop_stopped", "ReAct loop hit max iterations", max_iterations=self.max_iterations)

        return results
