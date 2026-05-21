"""
Stage-aware constraint reminders for long-running agent tasks.

In long-running tasks, the model's attention decays: constraints defined early
in a plan slip out of the context window. This module periodically re-injects
phase-specific guardrails into the most recent context, keeping the model
aligned without bloating system prompts.

Key insight (harness engineering):
  - Stable rules → system prompt (persistent, slow to decay)
  - Dynamic / phase-specific constraints → periodically injected into recent context
  - Without this, agents enter "rush mode": skipping steps, dropping citations,
    relaxing quality constraints as attention fades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Phase mapping — classify each tool action into a semantic phase
# ---------------------------------------------------------------------------

_PHASE_MAP: dict[str, str] = {
    "search_knowledge_base": "search",
    "web_search": "search",
    "fetch_webpage": "search",
    "read_document_detail": "read",
    "summarize_content": "read",
    "compare_concepts": "compare",
    "generate_research_report": "synthesize",
    "ask_user_clarification": "clarify",
}


def phase_for_action(action: str) -> str:
    return _PHASE_MAP.get(action, "execute")


# ---------------------------------------------------------------------------
# Stage constraint definitions
# ---------------------------------------------------------------------------


@dataclass
class StageConstraints:
    """Constraints partitioned by phase and scope.

    global_rules:      always active, injected in abbreviated form every N steps
    phase_rules:       phase_name -> list of constraint strings
    anti_rush_rules:   injected when rush detection fires
    """

    global_rules: list[str] = field(default_factory=list)
    phase_rules: dict[str, list[str]] = field(default_factory=dict)
    anti_rush_rules: list[str] = field(default_factory=list)


def default_stage_constraints() -> StageConstraints:
    """Build a sensible default constraint set for research tasks.

    These defaults encode the harness engineering principle: every phase has
    explicit guardrails that are re-injected into recent context so the model
    doesn't silently relax them as the task progresses.
    """
    return StageConstraints(
        global_rules=[
            "每个事实性观点必须标注引用 [来源N]，不允许无引用的断言",
            "不确定的信息要标注不确定性，不要假装确定",
            "如果检索不到相关信息，诚实说明而非编造",
            "注意信息时效性，优先使用最新资料",
        ],
        phase_rules={
            "search": [
                "搜索阶段：广泛收集信息，不要过早下结论",
                "使用多个查询角度搜索，避免单一关键词遗漏",
                "记录搜索结果的发表日期，标注时效性",
            ],
            "read": [
                "阅读阶段：仔细理解文档内容，提取关键信息",
                "注意区分作者观点和客观事实",
                "标注每个关键信息的来源段落",
            ],
            "compare": [
                "对比阶段：逐维度比较，不要跳过任何维度",
                "每个对比点都需要两边都有证据支撑",
                "避免选择性引用——优缺点都要呈现",
            ],
            "synthesize": [
                "综合阶段：基于前面所有步骤的产出，不要引入新信息",
                "报告结构要完整：背景 → 发现 → 分析 → 结论",
                "每个观点必须有来源支撑，不允许凭空总结",
                "检查是否有矛盾信息，如有矛盾要明确指出",
            ],
            "clarify": [
                "澄清阶段：问题要具体，提供选项引导用户快速决策",
                "不要一次问太多问题，聚焦最关键的不确定性",
            ],
        },
        anti_rush_rules=[
            "⚠️ 检测到可能的赶工模式——请放慢速度，确保每步质量",
            "⚠️ 不要跳过引用标注，每个观点都必须有来源",
            "⚠️ 输出内容不要缩水，保持之前的详细程度",
        ],
    )


# ---------------------------------------------------------------------------
# Rush detection
# ---------------------------------------------------------------------------


@dataclass
class RushSignal:
    """Heuristic rush / attention-decay detection result."""

    output_shrinking: bool = False
    citations_dropping: bool = False
    steps_accelerating: bool = False
    score: float = 0.0
    message: str = ""

    @property
    def is_rushing(self) -> bool:
        return self.score >= 0.5


class RushDetector:
    """Lightweight heuristic detector for attention decay / rush mode.

    Tracks output lengths and citation counts across steps. When the trend
    shows consistent decline, it signals that the agent may be "rushing"
    — producing shorter, less-cited outputs as attention fades.
    """

    def __init__(self, window_size: int = 4, shrink_ratio: float = 0.4):
        self.window_size = window_size
        self.shrink_ratio = shrink_ratio
        self.output_lengths: list[int] = []
        self.citation_counts: list[int] = []

    def record_step(self, output: str, citation_count: int = 0) -> None:
        self.output_lengths.append(len(output))
        self.citation_counts.append(citation_count)

    def detect(self) -> RushSignal:
        signal = RushSignal()
        if len(self.output_lengths) < self.window_size:
            return signal

        recent = self.output_lengths[-self.window_size:]
        earlier = self.output_lengths[-self.window_size * 2 : -self.window_size]
        if not earlier:
            return signal

        avg_recent = sum(recent) / len(recent)
        avg_earlier = sum(earlier) / len(earlier)
        if avg_earlier > 0 and avg_recent < avg_earlier * (1 - self.shrink_ratio):
            signal.output_shrinking = True
            signal.score += 0.4

        if len(self.citation_counts) >= self.window_size:
            recent_cite = self.citation_counts[-self.window_size:]
            earlier_cite = self.citation_counts[-self.window_size * 2 : -self.window_size]
            if earlier_cite:
                avg_recent_c = sum(recent_cite) / len(recent_cite)
                avg_earlier_c = sum(earlier_cite) / len(earlier_cite)
                if avg_earlier_c > 0 and avg_recent_c < avg_earlier_c * 0.5:
                    signal.citations_dropping = True
                    signal.score += 0.4

        if signal.output_shrinking and signal.citations_dropping:
            signal.score += 0.2

        signal.score = min(signal.score, 1.0)

        if signal.is_rushing:
            parts = []
            if signal.output_shrinking:
                parts.append("输出长度持续下降")
            if signal.citations_dropping:
                parts.append("引用数量持续下降")
            signal.message = "；".join(parts)
        return signal


# ---------------------------------------------------------------------------
# Stage reminder manager
# ---------------------------------------------------------------------------


@dataclass
class _PhaseTracker:
    """Internal tracker for phase transitions."""

    current_phase: str = ""
    phase_step_count: int = 0
    total_steps_executed: int = 0


class StageReminderManager:
    """Manages phase-aware constraint reminders across a plan execution.

    Two responsibilities:
      1. Periodically re-inject phase constraints into recent context
      2. Detect rush mode and escalate with anti-rush warnings

    Integration point:
      Before each step executes, the ReActLoop calls build_reminder(step) and
      injects the result into the step's invocation context. LLM-calling tool
      handlers prepend it to their prompt, keeping constraints in the model's
      attention window.
    """

    def __init__(self, constraints: StageConstraints | None = None,
                 reminder_interval: int = 2,
                 rush_detector: RushDetector | None = None):
        self.constraints = constraints or default_stage_constraints()
        self.reminder_interval = reminder_interval
        self.rush_detector = rush_detector or RushDetector()
        self._tracker = _PhaseTracker()

    def set_constraints(self, constraints: StageConstraints) -> None:
        self.constraints = constraints

    def record_step_result(self, output: str, citation_count: int = 0) -> None:
        self.rush_detector.record_step(output, citation_count)

    def build_reminder(self, step: dict[str, Any], step_results: list[Any] | None = None) -> str:
        """Build the reminder text to inject before executing `step`.

        Returns an empty string when no reminder is needed this step.
        """
        action = str(step.get("action", ""))
        phase = phase_for_action(action)
        self._tracker.total_steps_executed += 1

        # Track phase transitions
        phase_changed = phase != self._tracker.current_phase
        if phase_changed:
            self._tracker.current_phase = phase
            self._tracker.phase_step_count = 0
        self._tracker.phase_step_count += 1

        # Compute output lengths for rush detection if results available
        if step_results:
            self._record_from_results(step_results)

        rush = self.rush_detector.detect()

        # Decide whether to inject
        should_inject_full = phase_changed
        should_inject_brief = (
            self._tracker.total_steps_executed % self.reminder_interval == 0
        )
        should_inject_rush = rush.is_rushing

        if not (should_inject_full or should_inject_brief or should_inject_rush):
            return ""

        parts: list[str] = []
        step_id = step.get("step_id", "?")
        parts.append(f"## 执行状态提醒（第{step_id}步，阶段：{phase}）")

        if phase_changed:
            phase_constraints = self.constraints.phase_rules.get(phase, [])
            if phase_constraints:
                parts.append(f"**进入「{phase}」阶段，当前阶段注意事项：**")
                for rule in phase_constraints:
                    parts.append(f"- {rule}")

        if should_inject_brief and not phase_changed:
            parts.append("**全局约束（周期性重申）：**")
            for rule in self.constraints.global_rules:
                parts.append(f"- {rule}")

        if should_inject_rush and rush.message:
            parts.append(f"**{rush.message}**")
            for rule in self.constraints.anti_rush_rules:
                parts.append(rule)

        return "\n".join(parts)

    def _record_from_results(self, results: list[Any]) -> None:
        for r in results:
            output = getattr(r, "output", "") or ""
            data = getattr(r, "data", None) or {}
            citations = data.get("citations", {}) if isinstance(data, dict) else {}
            cite_count = len(citations) if isinstance(citations, dict) else 0
            self.rush_detector.record_step(output, cite_count)

    @property
    def current_phase(self) -> str:
        return self._tracker.current_phase

    @property
    def total_steps_executed(self) -> int:
        return self._tracker.total_steps_executed
