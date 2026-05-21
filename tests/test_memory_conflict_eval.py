import unittest
from pathlib import Path

from evals.memory_conflict_eval import run_memory_conflict_eval


ROOT = Path(__file__).resolve().parents[1]


class MemoryConflictEvalTests(unittest.TestCase):
    def test_memory_conflict_eval_cases_pass(self):
        report = run_memory_conflict_eval(ROOT / "evals" / "cases" / "memory_conflict_cases.jsonl")
        self.assertEqual(report["passed"], report["total"])


if __name__ == "__main__":
    unittest.main()
