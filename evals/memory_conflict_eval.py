from __future__ import annotations

import json
from pathlib import Path

from memory.long_term import LongTermMemory


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_memory_conflict_eval(cases_path: Path) -> dict:
    cases = load_jsonl(cases_path)
    results = []
    for case in cases:
        actual = LongTermMemory._looks_conflicting(case["existing"], case["incoming"])
        passed = actual == case["expected_conflict"]
        results.append({
            "id": case["id"],
            "category": case.get("category", ""),
            "passed": passed,
            "expected_conflict": case["expected_conflict"],
            "actual_conflict": actual,
        })

    passed_count = sum(1 for result in results if result["passed"])
    return {
        "suite": "memory_conflict",
        "total": len(results),
        "passed": passed_count,
        "accuracy": passed_count / len(results) if results else 0.0,
        "results": results,
    }
