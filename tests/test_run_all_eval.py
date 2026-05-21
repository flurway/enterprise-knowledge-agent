import asyncio
import unittest

from evals.run_all import run_all


class RunAllEvalTests(unittest.TestCase):
    def test_run_all_offline_suites_pass(self):
        summary = asyncio.run(run_all(no_report=True))
        self.assertTrue(summary["overall_passed"])
        self.assertEqual({suite["suite"] for suite in summary["suites"]}, {
            "security",
            "kb_retrieval",
            "rag_pipeline",
            "answer_validation",
            "memory_write",
            "memory_conflict",
            "memory_retrieval",
        })


if __name__ == "__main__":
    unittest.main()
