from __future__ import annotations

import json
from pathlib import Path

from memory.write_gate import MemoryWriteGate


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_memory_write_eval(cases_path: Path, min_response_chars: int = 40) -> dict:
    gate = MemoryWriteGate(min_response_chars=min_response_chars)
    cases = load_jsonl(cases_path)
    results = []
    for case in cases:
        decision = gate.decide(case["query"], case["result"], session_id=case.get("session_id", "eval"))
        checks = [decision.should_write == case["expected_should_write"]]
        if case.get("expected_memory_type"):
            checks.append(decision.memory_type == case["expected_memory_type"])
        if "expected_source_grounded" in case:
            checks.append(decision.metadata.get("source_grounded") == case["expected_source_grounded"])
        if "min_confidence" in case:
            checks.append(decision.confidence >= case["min_confidence"])
        if "max_confidence" in case:
            checks.append(decision.confidence <= case["max_confidence"])

        passed = all(checks)
        results.append({
            "id": case["id"],
            "passed": passed,
            "expected_should_write": case["expected_should_write"],
            "actual_should_write": decision.should_write,
            "memory_type": decision.memory_type,
            "confidence": decision.confidence,
            "reason": decision.reason,
        })

    passed_count = sum(1 for result in results if result["passed"])
    return {
        "suite": "memory_write",
        "total": len(results),
        "passed": passed_count,
        "accuracy": passed_count / len(results) if results else 0.0,
        "results": results,
    }
