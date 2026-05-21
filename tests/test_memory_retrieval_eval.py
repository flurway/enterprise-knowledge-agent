import unittest
from pathlib import Path

from evals.memory_retrieval_eval import run_memory_retrieval_eval


ROOT = Path(__file__).resolve().parents[1]


class MemoryRetrievalEvalTests(unittest.TestCase):
    def test_memory_retrieval_eval_cases_pass(self):
        report = run_memory_retrieval_eval(ROOT / "evals" / "cases" / "memory_retrieval_cases.jsonl")
        self.assertEqual(report["passed"], report["total"])


if __name__ == "__main__":
    unittest.main()
