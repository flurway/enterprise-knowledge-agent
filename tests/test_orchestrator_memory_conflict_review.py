import unittest
import sys
import types

from agent.state import AgentRun

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=None))

    fake_openai.AsyncOpenAI = FakeAsyncOpenAI
    sys.modules["openai"] = fake_openai

from agent.orchestrator import ResearchAgent


class FakeLongTermMemory:
    def __init__(self):
        self.resolution = None

    def get_conflict_reviews(self, status="unresolved", limit=1):
        return [{
            "conflict_index": 0,
            "conflict_type": "contradiction",
            "resolution_strategy": "manual_review",
            "existing_index": 1,
            "incoming_index": 2,
            "existing_content": "旧结论",
            "incoming_content": "新结论",
        }]

    def resolve_conflict(self, conflict_index, resolution):
        self.resolution = resolution
        return {"resolved": True, "conflict_index": conflict_index, "resolution": resolution}


class OrchestratorMemoryConflictReviewTests(unittest.TestCase):
    def test_parse_memory_conflict_resolution(self):
        self.assertEqual(ResearchAgent._parse_memory_conflict_resolution("采用新记忆"), "use_incoming")
        self.assertEqual(ResearchAgent._parse_memory_conflict_resolution("采用旧记忆"), "use_existing")
        self.assertEqual(ResearchAgent._parse_memory_conflict_resolution("保留两者"), "keep_both")
        self.assertIsNone(ResearchAgent._parse_memory_conflict_resolution("继续研究 RAG"))

    def test_try_resolve_memory_conflict_command(self):
        agent = ResearchAgent.__new__(ResearchAgent)
        agent.long_term_memory = FakeLongTermMemory()
        result = agent._try_resolve_memory_conflict_command("采用新记忆", AgentRun("s", "采用新记忆"))

        self.assertEqual(agent.long_term_memory.resolution, "use_incoming")
        self.assertEqual(result["intent"], "memory_conflict_review")
        self.assertTrue(result["memory_conflict_resolution"]["resolved"])

    def test_attach_conflict_aware_answer(self):
        agent = ResearchAgent.__new__(ResearchAgent)
        result = {"response": "原回答", "citations": {}}
        run = AgentRun("s", "q")
        agent._attach_conflict_aware_answer(result, [{
            "conflict_index": 0,
            "conflict_type": "contradiction",
            "resolution_strategy": "manual_review",
        }], run)

        self.assertTrue(result["needs_memory_confirmation"])
        self.assertEqual(result["memory_conflicts"][0]["conflict_index"], 0)
        self.assertIn("采用旧记忆", result["response"])


if __name__ == "__main__":
    unittest.main()
