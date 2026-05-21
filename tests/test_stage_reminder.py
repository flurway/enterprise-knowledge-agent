import unittest
import asyncio

from agent.stage_reminder import (
    StageConstraints,
    StageReminderManager,
    RushDetector,
    RushSignal,
    default_stage_constraints,
    phase_for_action,
)
from agent.tool_dispatcher import ToolInvocationContext
from agent.hooks import HookContext, HookDecision, HookManager
from agent.runtime import AgentRuntime, ReActLoop, PermissionMode, TodoStatus, RuntimeStepResult
from agent.state import AgentRun, StepStatus


class PhaseForActionTests(unittest.TestCase):
    def test_known_actions_map_to_expected_phases(self):
        self.assertEqual(phase_for_action("search_knowledge_base"), "search")
        self.assertEqual(phase_for_action("web_search"), "search")
        self.assertEqual(phase_for_action("fetch_webpage"), "search")
        self.assertEqual(phase_for_action("read_document_detail"), "read")
        self.assertEqual(phase_for_action("summarize_content"), "read")
        self.assertEqual(phase_for_action("compare_concepts"), "compare")
        self.assertEqual(phase_for_action("generate_research_report"), "synthesize")
        self.assertEqual(phase_for_action("ask_user_clarification"), "clarify")

    def test_unknown_action_falls_back_to_execute(self):
        self.assertEqual(phase_for_action("some_future_tool"), "execute")
        self.assertEqual(phase_for_action(""), "execute")


class DefaultStageConstraintsTests(unittest.TestCase):
    def test_all_expected_phases_have_rules(self):
        c = default_stage_constraints()
        for phase in ["search", "read", "compare", "synthesize", "clarify"]:
            self.assertIn(phase, c.phase_rules, f"missing phase: {phase}")
            self.assertIsInstance(c.phase_rules[phase], list)
            self.assertTrue(len(c.phase_rules[phase]) > 0, f"empty rules for phase: {phase}")

    def test_global_and_anti_rush_rules_present(self):
        c = default_stage_constraints()
        self.assertGreater(len(c.global_rules), 0)
        self.assertGreater(len(c.anti_rush_rules), 0)


class RushDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = RushDetector(window_size=2, shrink_ratio=0.4)

    def test_no_signal_when_below_window(self):
        self.detector.record_step("hello", 1)
        signal = self.detector.detect()
        self.assertEqual(signal.score, 0.0)
        self.assertFalse(signal.is_rushing)

    def test_no_signal_when_outputs_stable(self):
        for _ in range(4):
            self.detector.record_step("a" * 500, 3)
        signal = self.detector.detect()
        self.assertFalse(signal.is_rushing)

    def test_signal_when_output_shrinks_significantly(self):
        self.detector.record_step("a" * 1000, 5)
        self.detector.record_step("a" * 900, 4)
        self.detector.record_step("a" * 300, 2)
        self.detector.record_step("a" * 200, 1)
        signal = self.detector.detect()
        self.assertTrue(signal.output_shrinking)
        self.assertGreater(signal.score, 0.0)

    def test_signal_when_citations_drop(self):
        for _ in range(3):
            self.detector.record_step("a" * 500, 5)
        for _ in range(3):
            self.detector.record_step("a" * 500, 1)
        signal = self.detector.detect()
        self.assertTrue(signal.citations_dropping)
        self.assertGreater(signal.score, 0.0)

    def test_combined_signal_reaches_max_score(self):
        """Both shrinking and dropping should produce score near 1.0."""
        self.detector.record_step("a" * 1000, 5)
        self.detector.record_step("a" * 800, 4)
        self.detector.record_step("a" * 300, 1)
        self.detector.record_step("a" * 200, 0)
        signal = self.detector.detect()
        self.assertTrue(signal.is_rushing)
        self.assertGreater(signal.score, 0.5)
        self.assertIn("输出长度", signal.message)
        self.assertIn("引用", signal.message)

    def test_record_step_also_accepts_zero_citations(self):
        self.detector.record_step("output", 0)
        self.assertEqual(len(self.detector.citation_counts), 1)


class StageReminderManagerTests(unittest.TestCase):
    def setUp(self):
        self.constraints = StageConstraints(
            global_rules=["G1: cite sources", "G2: be honest"],
            phase_rules={
                "search": ["S1: search broadly", "S2: check dates"],
                "synthesize": ["SYN1: no new info", "SYN2: full structure"],
            },
            anti_rush_rules=["RUSH: slow down"],
        )

    def test_phase_transition_injects_full_constraints(self):
        mgr = StageReminderManager(constraints=self.constraints)
        step = {"step_id": 1, "action": "search_knowledge_base"}
        reminder = mgr.build_reminder(step)
        self.assertIn("S1: search broadly", reminder)
        self.assertIn("S2: check dates", reminder)

    def test_same_phase_no_interval_produces_empty(self):
        mgr = StageReminderManager(constraints=self.constraints, reminder_interval=10)
        mgr.build_reminder({"step_id": 1, "action": "search_knowledge_base"})
        reminder = mgr.build_reminder({"step_id": 2, "action": "web_search"})
        self.assertEqual(reminder, "")

    def test_periodic_interval_injects_global_rules(self):
        mgr = StageReminderManager(constraints=self.constraints, reminder_interval=2)
        mgr.build_reminder({"step_id": 1, "action": "search_knowledge_base"})  # phase change
        reminder = mgr.build_reminder({"step_id": 2, "action": "web_search"})  # same phase, interval=2
        self.assertIn("G1: cite sources", reminder)
        self.assertIn("G2: be honest", reminder)
        self.assertNotIn("S1", reminder)

    def test_phase_change_takes_priority_over_periodic(self):
        """When both phase change and periodic interval coincide, phase rules win."""
        mgr = StageReminderManager(constraints=self.constraints, reminder_interval=2)
        mgr.build_reminder({"step_id": 1, "action": "search_knowledge_base"})
        mgr.build_reminder({"step_id": 2, "action": "search_knowledge_base"})  # periodic
        # step 3: phase change to synthesize, also interval=2 hits
        reminder = mgr.build_reminder({"step_id": 3, "action": "generate_research_report"})
        self.assertIn("SYN1", reminder)
        self.assertIn("SYN2", reminder)

    def test_record_step_result_feeds_rush_detector(self):
        mgr = StageReminderManager(constraints=self.constraints, reminder_interval=10)
        # RushDetector needs 2*window_size samples (default window=4, so 8) to
        # compare "recent" vs "earlier" windows.
        # First 4: stable, large outputs
        for _ in range(4):
            mgr.record_step_result("a" * 1000, 5)
        # Next 4: shrinking outputs + dropping citations
        mgr.record_step_result("a" * 600, 3)
        mgr.record_step_result("a" * 400, 2)
        mgr.record_step_result("a" * 200, 1)
        mgr.record_step_result("a" * 150, 0)

        rush = mgr.rush_detector.detect()
        self.assertTrue(rush.is_rushing)

    def test_anti_rush_injected_when_rushing_detected(self):
        mgr = StageReminderManager(constraints=self.constraints, reminder_interval=1)
        # Prime the rush detector
        for _ in range(3):
            mgr.record_step_result("a" * 1000, 5)
        for _ in range(3):
            mgr.record_step_result("a" * 200, 0)

        step = {"step_id": 7, "action": "search_knowledge_base"}
        reminder = mgr.build_reminder(step)
        self.assertIn("RUSH", reminder)

    def test_empty_constraints_produces_minimal_reminder(self):
        empty = StageConstraints(global_rules=[], phase_rules={}, anti_rush_rules=[])
        mgr = StageReminderManager(constraints=empty)
        step = {"step_id": 1, "action": "search_knowledge_base"}
        reminder = mgr.build_reminder(step)
        # Still contains header even with empty rules
        self.assertIn("执行状态提醒", reminder)

    def test_set_constraints_replaces_existing(self):
        mgr = StageReminderManager(constraints=self.constraints)
        new_constraints = StageConstraints(
            global_rules=["NEW: only this"],
            phase_rules={"search": ["NEW: search rule"]},
            anti_rush_rules=[],
        )
        mgr.set_constraints(new_constraints)
        # Step 1: phase change → only phase rules shown (not global rules)
        reminder1 = mgr.build_reminder({"step_id": 1, "action": "search_knowledge_base"})
        self.assertIn("NEW: search rule", reminder1)
        self.assertNotIn("G1", reminder1)
        # Step 2: same phase, interval=2 → periodic global rules shown
        reminder2 = mgr.build_reminder({"step_id": 2, "action": "web_search"})
        self.assertIn("NEW: only this", reminder2)
        self.assertNotIn("NEW: search rule", reminder2)

    def test_total_steps_tracks_correctly(self):
        mgr = StageReminderManager(constraints=self.constraints)
        for i in range(5):
            mgr.build_reminder({"step_id": i + 1, "action": "web_search"})
        self.assertEqual(mgr.total_steps_executed, 5)

    def test_current_phase_tracks_phase_transitions(self):
        mgr = StageReminderManager(constraints=self.constraints)
        mgr.build_reminder({"step_id": 1, "action": "search_knowledge_base"})
        self.assertEqual(mgr.current_phase, "search")
        mgr.build_reminder({"step_id": 2, "action": "generate_research_report"})
        self.assertEqual(mgr.current_phase, "synthesize")


class StageReminderIntegrationTests(unittest.TestCase):
    """Verify stage_reminder flows end-to-end through ToolInvocationContext."""

    def test_tool_invocation_context_carries_stage_reminder(self):
        ctx = ToolInvocationContext(
            action="summarize_content",
            params={"focus": "test"},
            dep_context="prior output",
            stage_reminder="## 执行状态提醒（第3步，阶段：read）\n**全局约束（周期性重申）：**\n- 每个事实性观点必须标注引用",
        )
        self.assertIn("执行状态提醒", ctx.stage_reminder)
        self.assertIn("全局约束", ctx.stage_reminder)
        self.assertEqual(ctx.action, "summarize_content")
        self.assertEqual(ctx.dep_context, "prior output")

    def test_tool_invocation_context_defaults_to_empty_reminder(self):
        ctx = ToolInvocationContext(action="search_knowledge_base", params={})
        self.assertEqual(ctx.stage_reminder, "")


class PreStepInjectHookTests(unittest.TestCase):
    def test_pre_step_inject_concatenates_hook_outputs(self):
        manager = HookManager()
        manager.register_pre_step_inject(
            lambda ctx: "INJECT_A" if ctx.payload.get("step_id") == 1 else None
        )
        manager.register_pre_step_inject(
            lambda ctx: "INJECT_B" if ctx.payload.get("step_id") == 1 else None
        )

        result = manager.run_pre_step_inject(HookContext(
            event="pre_step_inject",
            payload={"step_id": 1, "action": "search_knowledge_base"},
        ))
        self.assertIn("INJECT_A", result)
        self.assertIn("INJECT_B", result)

    def test_pre_step_inject_returns_empty_when_no_hooks_fire(self):
        manager = HookManager()
        manager.register_pre_step_inject(lambda ctx: None)
        result = manager.run_pre_step_inject(HookContext(
            event="pre_step_inject",
            payload={"step_id": 99},
        ))
        self.assertEqual(result, "")

    def test_pre_step_inject_does_not_block(self):
        """pre_step_inject is context augmentation, not blocking — it should
        never affect the allowed/blocked decision flow."""
        manager = HookManager()
        manager.register_pre_step_inject(lambda ctx: "some reminder text")
        manager.register_pre_tool_use(lambda ctx: HookDecision(False, "blocked by pre_tool"))

        inject_result = manager.run_pre_step_inject(HookContext(
            event="pre_step_inject", payload={},
        ))
        self.assertEqual(inject_result, "some reminder text")

        block_result = manager.run_pre_tool_use(HookContext(
            event="pre_tool_use", payload={},
        ))
        self.assertFalse(block_result.allowed)


class StageReminderInReActLoopTests(unittest.TestCase):
    """Verify that StageReminderManager is wired into AgentRuntime and ReActLoop."""

    def test_agent_runtime_has_stage_reminder_by_default(self):
        run = AgentRun(session_id="s", user_message="q")
        runtime = AgentRuntime(run)
        self.assertIsNotNone(runtime.stage_reminder)
        self.assertTrue(hasattr(runtime.stage_reminder, "build_reminder"))

    def test_react_loop_injects_stage_reminder_into_step(self):
        async def run():
            run_obj = AgentRun(session_id="s", user_message="q")
            runtime = AgentRuntime(run_obj)
            runtime.set_permission_mode(PermissionMode.AUTO)
            plan = {
                "steps": [
                    {"step_id": 1, "action": "search_knowledge_base", "description": "Search", "depends_on": []},
                    {"step_id": 2, "action": "generate_research_report", "description": "Report", "depends_on": [1]},
                ]
            }
            runtime.set_plan(plan)
            runtime.load_plan_as_todos(plan)

            captured_reminders = []

            async def execute(step):
                reminder = step.get("_stage_reminder", "")
                captured_reminders.append(reminder)
                return RuntimeStepResult(step["step_id"], step["action"], True, output="ok")

            results = await ReActLoop(runtime).run_plan_todos(
                {step["step_id"]: step for step in plan["steps"]},
                execute,
            )
            return captured_reminders, results

        captured, results = asyncio.run(run())
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))
        # Step 1 should have a phase transition reminder (first step -> search phase)
        self.assertIn("执行状态提醒", captured[0])
        self.assertIn("search", captured[0].lower())


if __name__ == "__main__":
    unittest.main()
