"""
Runtime hook manager.

Hooks are deterministic lifecycle extension points around agent actions. They
make cross-cutting controls such as permission checks, security scans, trace
events, answer validation, and future MCP adapters easier to reason about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HookDecision:
    allowed: bool = True
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookContext:
    event: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


PreToolHook = Callable[[HookContext], HookDecision | None]
PostToolHook = Callable[[HookContext], HookDecision | None]
AnswerReadyHook = Callable[[HookContext], HookDecision | None]
PreStepInjectHook = Callable[[HookContext], str | None]


class HookManager:
    def __init__(self):
        self.pre_tool_use_hooks: list[PreToolHook] = []
        self.post_tool_use_hooks: list[PostToolHook] = []
        self.answer_ready_hooks: list[AnswerReadyHook] = []
        self.pre_step_inject_hooks: list[PreStepInjectHook] = []

    def register_pre_tool_use(self, hook: PreToolHook) -> None:
        self.pre_tool_use_hooks.append(hook)

    def register_post_tool_use(self, hook: PostToolHook) -> None:
        self.post_tool_use_hooks.append(hook)

    def register_answer_ready(self, hook: AnswerReadyHook) -> None:
        self.answer_ready_hooks.append(hook)

    def register_pre_step_inject(self, hook: PreStepInjectHook) -> None:
        self.pre_step_inject_hooks.append(hook)

    def run_pre_tool_use(self, context: HookContext) -> HookDecision:
        return self._run_until_blocked(self.pre_tool_use_hooks, context)

    def run_post_tool_use(self, context: HookContext) -> HookDecision:
        return self._run_until_blocked(self.post_tool_use_hooks, context)

    def run_answer_ready(self, context: HookContext) -> HookDecision:
        return self._run_until_blocked(self.answer_ready_hooks, context)

    def run_pre_step_inject(self, context: HookContext) -> str:
        """Run all pre-step-inject hooks and return concatenated inject text.

        Unlike other hook types, pre_step_inject does not block — it returns
        context-augmentation text to be prepended to the step prompt.
        """
        parts: list[str] = []
        for hook in self.pre_step_inject_hooks:
            text = hook(context)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _run_until_blocked(hooks: list[Callable[[HookContext], HookDecision | None]], context: HookContext) -> HookDecision:
        merged_metadata: dict[str, Any] = {}
        for hook in hooks:
            decision = hook(context)
            if decision is None:
                continue
            merged_metadata.update(decision.metadata)
            if not decision.allowed:
                return HookDecision(False, decision.reason, merged_metadata)
        return HookDecision(True, "All hooks allowed", merged_metadata)


def trace_hook(event_name: str, run_getter: Callable[[], Any]) -> Callable[[HookContext], HookDecision]:
    def hook(context: HookContext) -> HookDecision:
        run = run_getter()
        if run is not None:
            run.add_event(event_name, f"Hook event: {event_name}", **context.payload)
        return HookDecision(True)

    return hook

