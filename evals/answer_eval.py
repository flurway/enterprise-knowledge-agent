from __future__ import annotations

import json
from pathlib import Path

from agent.answer_validator import validate_answer


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_answer_eval(cases_path: Path) -> dict:
    cases = load_cases(cases_path)
    results = []
    for case in cases:
        validation = validate_answer(case["answer"], case.get("citations", {}))
        passed = validation.ok == case["expected_ok"]
        results.append({
            "id": case["id"],
            "passed": passed,
            "expected_ok": case["expected_ok"],
            "actual_ok": validation.ok,
            "validation": validation.to_dict(),
        })
    total = len(results)
    return {
        "suite": "answer_validation",
        "total": total,
        "passed": sum(1 for r in results if r["passed"]),
        "accuracy": sum(1 for r in results if r["passed"]) / max(total, 1),
        "results": results,
    }

