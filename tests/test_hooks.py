import unittest

from agent.hooks import HookContext, HookDecision, HookManager


class HookManagerTest(unittest.TestCase):
    def test_pre_tool_hooks_merge_metadata_when_allowed(self):
        manager = HookManager()

        manager.register_pre_tool_use(lambda context: HookDecision(True, metadata={"first": context.event}))
        manager.register_pre_tool_use(lambda context: HookDecision(True, metadata={"second": context.payload["tool"]}))

        decision = manager.run_pre_tool_use(HookContext(
            event="pre_tool_use",
            payload={"tool": "search_knowledge_base"},
        ))

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.metadata["first"], "pre_tool_use")
        self.assertEqual(decision.metadata["second"], "search_knowledge_base")

    def test_pre_tool_hooks_stop_on_block(self):
        manager = HookManager()
        calls = []

        def block(context):
            calls.append("block")
            return HookDecision(False, "requires approval", {"risk": "high"})

        def after_block(context):
            calls.append("after_block")
            return HookDecision(True)

        manager.register_pre_tool_use(block)
        manager.register_pre_tool_use(after_block)

        decision = manager.run_pre_tool_use(HookContext(event="pre_tool_use", payload={}))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "requires approval")
        self.assertEqual(decision.metadata["risk"], "high")
        self.assertEqual(calls, ["block"])

    def test_answer_ready_hooks_use_same_blocking_semantics(self):
        manager = HookManager()
        manager.register_answer_ready(lambda context: HookDecision(
            allowed="answer_validation" in context.payload,
            reason="missing answer validation",
        ))

        blocked = manager.run_answer_ready(HookContext(
            event="answer_ready",
            payload={"response_preview": "hello"},
        ))
        allowed = manager.run_answer_ready(HookContext(
            event="answer_ready",
            payload={"response_preview": "hello", "answer_validation": {"ok": True}},
        ))

        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, "missing answer validation")
        self.assertTrue(allowed.allowed)


if __name__ == "__main__":
    unittest.main()
