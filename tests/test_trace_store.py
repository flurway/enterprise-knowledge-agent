import os
import tempfile
import unittest

from agent.state import AgentRun, ToolCallRecord
from agent.trace_store import AgentTraceStore


class TraceStoreTests(unittest.TestCase):
    def test_save_list_and_get_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "traces.sqlite3")
            store = AgentTraceStore(db_path)

            run = AgentRun(session_id="s1", user_message="hello")
            run.intent = "chitchat"
            run.refined_query = "hello"
            run.add_event("intent_classified", "Intent classified as chitchat")
            run.record_tool_call(ToolCallRecord(
                tool_name="search_knowledge_base",
                step_id=1,
                success=True,
                latency_ms=12,
                policy_decision="allowed",
            ))
            run.finish(True)

            store.save_run(run)

            runs = store.list_runs(limit=5)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], run.run_id)
            self.assertEqual(runs[0]["tool_call_count"], 1)

            trace = store.get_run(run.run_id)
            self.assertIsNotNone(trace)
            self.assertEqual(trace["run_id"], run.run_id)
            self.assertEqual(trace["session_id"], "s1")
            self.assertEqual(trace["tool_calls"][0]["tool_name"], "search_knowledge_base")

    def test_list_runs_can_filter_by_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentTraceStore(os.path.join(tmpdir, "traces.sqlite3"))
            run1 = AgentRun(session_id="a", user_message="one")
            run1.finish(True)
            run2 = AgentRun(session_id="b", user_message="two")
            run2.finish(True)
            store.save_run(run1)
            store.save_run(run2)

            runs = store.list_runs(session_id="a")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["session_id"], "a")


if __name__ == "__main__":
    unittest.main()

