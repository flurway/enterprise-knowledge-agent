import unittest
import asyncio

from agent.runtime import AgentRuntime, PermissionMode, ReActLoop, RuntimeMode, RuntimeNode, RuntimeStepResult, TodoStatus
from agent.state import AgentRun, FailureType, StepStatus


class AgentRuntimeTests(unittest.TestCase):
    def test_enter_node_records_event(self):
        run = AgentRun(session_id="s", user_message="q")
        runtime = AgentRuntime(run)
        runtime.enter_node(RuntimeNode.CLASSIFY, "Classify")
        self.assertEqual(run.events[-1].event_type, "node_entered")
        self.assertEqual(run.events[-1].metadata["node"], "classify")

    def test_validate_plan_detects_duplicate_missing_dep_and_unknown_action(self):
        run = AgentRun(session_id="s", user_message="q")
        runtime = AgentRuntime(run)
        errors = runtime.validate_plan({
            "steps": [
                {"step_id": 1, "action": "search_knowledge_base", "depends_on": [99]},
                {"step_id": 1, "action": "unknown_tool", "depends_on": []},
            ]
        })
        self.assertTrue(any("duplicate step_id" in error for error in errors))
        self.assertTrue(any("unknown action" in error for error in errors))
        self.assertTrue(any("depends on missing step" in error for error in errors))

    def test_step_dependency_and_status_lifecycle(self):
        run = AgentRun(session_id="s", user_message="q")
        runtime = AgentRuntime(run)
        runtime.set_plan({
            "steps": [
                {"step_id": 1, "action": "search_knowledge_base", "depends_on": []},
                {"step_id": 2, "action": "generate_research_report", "depends_on": [1]},
            ]
        })

        can_execute, unmet = runtime.can_execute_step(2)
        self.assertFalse(can_execute)
        self.assertEqual(unmet, [1])

        runtime.mark_step_running(1)
        self.assertEqual(run.steps[0].status, StepStatus.RUNNING)
        runtime.mark_step_finished(1, True)
        self.assertEqual(run.steps[0].status, StepStatus.SUCCEEDED)

        can_execute, unmet = runtime.can_execute_step(2)
        self.assertTrue(can_execute)
        self.assertEqual(unmet, [])

    def test_failure_type_classification(self):
        run = AgentRun(session_id="s", user_message="q")
        runtime = AgentRuntime(run)
        runtime.set_plan({"steps": [{"step_id": 1, "action": "web_search", "depends_on": []}]})
        runtime.mark_step_running(1)
        runtime.mark_step_finished(1, False, "Policy blocked tool call: no")
        self.assertEqual(run.steps[0].status, StepStatus.FAILED)
        self.assertEqual(run.steps[0].failure_type, FailureType.POLICY_BLOCKED)

    def test_mode_permission_and_todo_flow(self):
        run = AgentRun(session_id="s", user_message="q")
        runtime = AgentRuntime(run)
        runtime.set_mode(RuntimeMode.PLAN, "planning first")
        runtime.set_permission_mode(PermissionMode.PLAN_ONLY, "no tools")
        todo = runtime.add_todo("Search docs")

        self.assertEqual(runtime.mode, RuntimeMode.PLAN)
        self.assertEqual(runtime.permission_mode, PermissionMode.PLAN_ONLY)
        self.assertEqual(runtime.next_pending_todo().todo_id, todo.todo_id)

        allowed, reason = runtime.check_permission_for_tool("search_knowledge_base")
        self.assertFalse(allowed)
        self.assertIn("Plan-only", reason)

        decision = runtime.decide_next()
        self.assertTrue(decision.done)
        self.assertEqual(decision.action, "finish")

        runtime.set_mode(RuntimeMode.EXECUTE, "execute now")
        runtime.set_permission_mode(PermissionMode.AUTO, "allowed tools")
        allowed, _ = runtime.check_permission_for_tool("search_knowledge_base")
        self.assertTrue(allowed)
        runtime.update_todo(todo.todo_id, TodoStatus.IN_PROGRESS)
        self.assertEqual(todo.status, TodoStatus.IN_PROGRESS)

    def test_read_only_permission_uses_tool_registry_scopes(self):
        run = AgentRun(session_id="s", user_message="q")
        runtime = AgentRuntime(run)
        runtime.set_permission_mode(PermissionMode.READ_ONLY)

        allowed, _ = runtime.check_permission_for_tool("code_search")
        self.assertTrue(allowed)

        allowed, reason = runtime.check_permission_for_tool("run_tests")
        self.assertFalse(allowed)
        self.assertIn("Read-only", reason)

        allowed, reason = runtime.check_permission_for_tool("code_patch")
        self.assertFalse(allowed)
        self.assertIn("Read-only", reason)

    def test_load_plan_as_todos(self):
        run = AgentRun(session_id="s", user_message="q")
        runtime = AgentRuntime(run)
        runtime.load_plan_as_todos({
            "steps": [
                {"step_id": 1, "action": "search_knowledge_base", "description": "Search KB"},
                {"step_id": 2, "action": "generate_research_report", "description": "Write report"},
            ]
        })
        self.assertEqual(len(runtime.todos), 2)
        self.assertEqual(runtime.todos[0].metadata["step_id"], 1)
        self.assertEqual(runtime.todos[1].metadata["action"], "generate_research_report")

    def test_react_loop_executes_plan_todos(self):
        async def run():
            run = AgentRun(session_id="s", user_message="q")
            runtime = AgentRuntime(run)
            runtime.set_permission_mode(PermissionMode.AUTO)
            plan = {
                "steps": [
                    {"step_id": 1, "action": "search_knowledge_base", "description": "Search", "depends_on": []},
                    {"step_id": 2, "action": "generate_research_report", "description": "Report", "depends_on": [1]},
                ]
            }
            runtime.set_plan(plan)
            runtime.load_plan_as_todos(plan)

            async def execute(step):
                return RuntimeStepResult(step["step_id"], step["action"], True, output="ok")

            results = await ReActLoop(runtime).run_plan_todos(
                {step["step_id"]: step for step in plan["steps"]},
                execute,
            )
            return runtime, results

        runtime, results = asyncio.run(run())
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.success for result in results))
        self.assertTrue(all(todo.status == TodoStatus.DONE for todo in runtime.todos))
        self.assertTrue(all(step.status == StepStatus.SUCCEEDED for step in runtime.run.steps))

    def test_react_loop_blocks_when_permission_denied(self):
        async def run():
            run = AgentRun(session_id="s", user_message="q")
            runtime = AgentRuntime(run)
            runtime.set_permission_mode(PermissionMode.PLAN_ONLY)
            plan = {"steps": [{"step_id": 1, "action": "search_knowledge_base", "description": "Search", "depends_on": []}]}
            runtime.set_plan(plan)
            runtime.load_plan_as_todos(plan)

            async def execute(step):
                raise AssertionError("execute should not be called")

            results = await ReActLoop(runtime).run_plan_todos({1: plan["steps"][0]}, execute)
            return runtime, results

        runtime, results = asyncio.run(run())
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertEqual(runtime.todos[0].status, TodoStatus.BLOCKED)
        self.assertEqual(runtime.run.steps[0].status, StepStatus.SKIPPED)


if __name__ == "__main__":
    unittest.main()
