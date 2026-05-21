from __future__ import annotations

import json
from pathlib import Path

from memory.retrieval_budget import MemoryRetrievalBudget


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_memory_retrieval_eval(cases_path: Path) -> dict:
    cases = load_jsonl(cases_path)
    results = []
    for case in cases:
        budget = MemoryRetrievalBudget(max_items=case.get("max_items", 3))
        selected = budget.select(case["memories"])
        actual_ids = [memory.get("id") for memory in selected]
        passed = actual_ids == case["expected_ids"]
        results.append({
            "id": case["id"],
            "passed": passed,
            "expected_ids": case["expected_ids"],
            "actual_ids": actual_ids,
        })

    passed_count = sum(1 for result in results if result["passed"])
    return {
        "suite": "memory_retrieval",
        "total": len(results),
        "passed": passed_count,
        "accuracy": passed_count / len(results) if results else 0.0,
        "results": results,
    }
