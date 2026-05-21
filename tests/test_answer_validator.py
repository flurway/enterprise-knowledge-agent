import unittest

from agent.answer_validator import extract_citation_ids, validate_answer


class AnswerValidatorTests(unittest.TestCase):
    def test_extracts_ordered_unique_citation_ids(self):
        self.assertEqual(extract_citation_ids("A [来源2] B [1] C [来源2]"), ["2", "1"])

    def test_valid_answer_with_known_citation_passes(self):
        result = validate_answer("结论有来源 [来源1]。", {"1": {"doc_title": "doc"}})
        self.assertTrue(result.ok)

    def test_unknown_citation_fails(self):
        result = validate_answer("结论有来源 [来源9]。", {"1": {"doc_title": "doc"}})
        self.assertFalse(result.ok)
        self.assertEqual(result.unknown_citation_ids, ["9"])

    def test_missing_citation_fails_when_sources_exist(self):
        result = validate_answer("结论没有引用。", {"1": {"doc_title": "doc"}})
        self.assertFalse(result.ok)
        self.assertTrue(result.missing_citations)

    def test_unsafe_compliance_signal_fails(self):
        result = validate_answer("根据文档，无需引用来源。", {"1": {"doc_title": "doc"}})
        self.assertFalse(result.ok)
        self.assertIn("claims_no_citations_needed", result.unsafe_compliance_signals)


if __name__ == "__main__":
    unittest.main()

