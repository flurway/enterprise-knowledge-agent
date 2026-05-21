import asyncio
import sys
import types
import unittest

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=None))

    fake_openai.AsyncOpenAI = FakeAsyncOpenAI
    sys.modules["openai"] = fake_openai

from agent.langgraph_runtime import LangGraphResearchRuntime, ResearchGraphState


class FakeMemory:
    def __init__(self):
        self.turns = []
        self.summary_threshold = 10

    def add_turn(self, role, content, metadata=None):
        self.turns.append((role, content, metadata or {}))


class FakeWorkflow:
    def __init__(self):
        self.state_schema = None
        self.handlers = None

    def build(self, handlers):
        self.handlers = handlers
        return FakeCompiledGraph(handlers)


class FakeCompiledGraph:
    def __init__(self, handlers):
        self.handlers = handlers

    async def ainvoke(self, state):
        state.update(await self.handlers["classify"](state))
        state.update(await self.handlers["finish"](state))
        return state


class FakeAgent:
    def __init__(self):
        self.memory = FakeMemory()

    def get_or_create_session(self, session_id):
        return self.memory

    def _try_resolve_memory_conflict_command(self, user_message, run):
        return {"response": "resolved", "intent": "memory_conflict_review", "citations": {}}

    async def _save_explicit_user_memory_candidates(self, user_message, session_id, run):
        raise AssertionError("conflict command should short-circuit before memory extraction")

    def _finish_result(self, result, run):
        run.finish(True)
        result["run_id"] = run.run_id
        return result


class LangGraphResearchRuntimeTests(unittest.TestCase):
    def test_default_workflow_uses_full_research_graph_state(self):
        runtime = LangGraphResearchRuntime(FakeAgent())
        self.assertIs(runtime.workflow.state_schema, ResearchGraphState)

    def test_chat_uses_graph_handlers_and_finishes_short_circuit_result(self):
        workflow = FakeWorkflow()
        runtime = LangGraphResearchRuntime(FakeAgent(), workflow=workflow)

        result = asyncio.run(runtime.chat("采用新记忆", session_id="s"))

        self.assertEqual(result["response"], "resolved")
        self.assertEqual(result["intent"], "memory_conflict_review")
        self.assertIn("run_id", result)
        self.assertIn("classify", workflow.handlers)
        self.assertIn("finish", workflow.handlers)


if __name__ == "__main__":
    unittest.main()
