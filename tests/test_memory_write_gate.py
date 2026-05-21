import unittest

from memory.write_gate import MemoryWriteGate


class MemoryWriteGateTest(unittest.TestCase):
    def test_skips_chitchat(self):
        decision = MemoryWriteGate().decide(
            "你好",
            {"intent": "chitchat", "response": "你好，很高兴见到你。", "citations": {}},
        )

        self.assertFalse(decision.should_write)
        self.assertIn("chitchat", decision.reason)

    def test_skips_clarification(self):
        decision = MemoryWriteGate().decide(
            "研究一下",
            {"needs_clarification": True, "response": "你想研究哪个方向？"},
        )

        self.assertFalse(decision.should_write)
        self.assertIn("clarification", decision.reason)

    def test_writes_grounded_research_summary(self):
        decision = MemoryWriteGate(min_response_chars=20).decide(
            "RAG 如何防注入？",
            {
                "intent": "direct_search",
                "response": "RAG 外部文档应该作为不可信证据处理，并在工具调用前执行 policy 检查。",
                "citations": {"1": {"doc_title": "prompt_injection_defense.md"}},
                "answer_validation": {"ok": True},
            },
            session_id="s1",
        )

        self.assertTrue(decision.should_write)
        self.assertEqual(decision.memory_type, "research_summary")
        self.assertGreater(decision.confidence, 0.8)
        self.assertTrue(decision.metadata["source_grounded"])
        self.assertEqual(decision.metadata["session_id"], "s1")

    def test_skips_invalid_grounded_answer(self):
        decision = MemoryWriteGate(min_response_chars=20).decide(
            "RAG 如何防注入？",
            {
                "intent": "direct_search",
                "response": "RAG 外部文档应该作为不可信证据处理，并在工具调用前执行 policy 检查。",
                "citations": {"1": {"doc_title": "prompt_injection_defense.md"}},
                "answer_validation": {"ok": False, "issues": ["missing citation"]},
            },
        )

        self.assertFalse(decision.should_write)
        self.assertIn("validation failed", decision.reason)

    def test_ungrounded_fallback_is_low_confidence_episode(self):
        decision = MemoryWriteGate(min_response_chars=20).decide(
            "解释一下 TCP",
            {
                "intent": "direct_search",
                "response": "TCP 是面向连接的传输层协议，提供可靠传输、拥塞控制和流量控制。",
                "citations": {},
            },
        )

        self.assertTrue(decision.should_write)
        self.assertEqual(decision.memory_type, "episode")
        self.assertLess(decision.confidence, 0.5)
        self.assertFalse(decision.metadata["source_grounded"])


if __name__ == "__main__":
    unittest.main()
