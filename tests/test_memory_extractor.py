import unittest

from memory.extractor import ExplicitMemoryExtractor


class ExplicitMemoryExtractorTest(unittest.TestCase):
    def test_extracts_chinese_preference(self):
        candidates = ExplicitMemoryExtractor().extract("我喜欢先看结论再看细节", session_id="s1")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "preference")
        self.assertIn("先看结论", candidates[0].content)
        self.assertEqual(candidates[0].metadata["session_id"], "s1")

    def test_extracts_chinese_constraint(self):
        candidates = ExplicitMemoryExtractor().extract("以后请都用中文回答，不要联网")

        types = {candidate.memory_type for candidate in candidates}
        contents = "\n".join(candidate.content for candidate in candidates)

        self.assertIn("constraint", types)
        self.assertIn("用中文回答", contents)
        self.assertIn("不要联网", contents)

    def test_does_not_infer_from_ordinary_question(self):
        candidates = ExplicitMemoryExtractor().extract("RAG 的 chunking 策略有哪些？")

        self.assertEqual(candidates, [])

    def test_extracts_english_preference(self):
        candidates = ExplicitMemoryExtractor().extract("I prefer concise answers with examples.")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "preference")
        self.assertIn("concise answers", candidates[0].content)


if __name__ == "__main__":
    unittest.main()
