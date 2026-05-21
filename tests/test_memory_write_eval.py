import unittest
from pathlib import Path

from evals.memory_write_eval import run_memory_write_eval


ROOT = Path(__file__).resolve().parents[1]


class MemoryWriteEvalTest(unittest.TestCase):
    def test_memory_write_cases_pass(self):
        report = run_memory_write_eval(ROOT / "evals" / "cases" / "memory_write_cases.jsonl")
        self.assertEqual(report["passed"], report["total"])
        self.assertEqual(report["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
