import unittest
from pathlib import Path

from evals.kb_eval import run_kb_eval


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeBaseEvalTests(unittest.TestCase):
    def test_fixture_kb_retrieval_cases_are_top1(self):
        report = run_kb_eval(
            kb_dir=ROOT / "evals" / "fixtures" / "knowledge_base",
            cases_path=ROOT / "evals" / "cases" / "kb_retrieval_cases.jsonl",
            top_k=3,
        )
        self.assertEqual(report["top1_accuracy"], 1.0)
        self.assertEqual(report["topk_recall"], 1.0)
        self.assertEqual(report["must_include_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

