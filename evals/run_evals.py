#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.metrics import accuracy, format_percent
from agent.policies import ToolPolicy
from security.injection_detector import scan_text
from security.untrusted_flow import suspicious_dependency_ids
from security.url_safety import check_response_safety, check_url_safety


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_security_suite() -> dict:
    cases = load_jsonl(ROOT / "evals" / "cases" / "security_cases.jsonl")
    results = []
    for case in cases:
        if case["type"] == "prompt_injection":
            scan = scan_text(case["text"])
            passed = scan.is_suspicious == case["expected_suspicious"]
            results.append({
                "id": case["id"],
                "type": case["type"],
                "passed": passed,
                "expected": case["expected_suspicious"],
                "actual": scan.is_suspicious,
                "detail": scan.summary(),
            })
        elif case["type"] == "url_safety":
            decision = check_url_safety(case["url"])
            passed = decision.allowed == case["expected_allowed"]
            results.append({
                "id": case["id"],
                "type": case["type"],
                "passed": passed,
                "expected": case["expected_allowed"],
                "actual": decision.allowed,
                "detail": decision.reason,
            })
        elif case["type"] == "response_safety":
            decision = check_response_safety(
                content_type=case.get("content_type", ""),
                content_length=case.get("content_length", ""),
            )
            passed = decision.allowed == case["expected_allowed"]
            results.append({
                "id": case["id"],
                "type": case["type"],
                "passed": passed,
                "expected": case["expected_allowed"],
                "actual": decision.allowed,
                "detail": decision.reason,
            })
        elif case["type"] == "untrusted_flow":
            execution_context = {
                item["step_id"]: SimpleNamespace(data=item.get("data", {}))
                for item in case.get("execution_context", [])
            }
            suspicious_ids = suspicious_dependency_ids(execution_context, case.get("depends_on", []))
            policy_decision = ToolPolicy().check(
                case["action"],
                case.get("params", {}),
                triggered_by_untrusted_context=bool(suspicious_ids),
            )
            expected_ids = case.get("expected_suspicious_dependency_ids", [])
            expected_allowed = case.get("expected_policy_allowed", True)
            passed = suspicious_ids == expected_ids and policy_decision.allowed == expected_allowed
            results.append({
                "id": case["id"],
                "type": case["type"],
                "passed": passed,
                "expected": {
                    "suspicious_dependency_ids": expected_ids,
                    "policy_allowed": expected_allowed,
                },
                "actual": {
                    "suspicious_dependency_ids": suspicious_ids,
                    "policy_allowed": policy_decision.allowed,
                },
                "detail": policy_decision.reason,
            })
    return {
        "suite": "security",
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "accuracy": accuracy([r["passed"] for r in results]),
        "results": results,
    }


def write_report(report: dict) -> Path:
    report_dir = ROOT / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"{report['suite']}_{stamp}.md"
    lines = [
        f"# Eval Report: {report['suite']}",
        "",
        f"- Total: {report['total']}",
        f"- Passed: {report['passed']}",
        f"- Accuracy: {format_percent(report['accuracy'])}",
        "",
        "| ID | Type | Passed | Expected | Actual | Detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(
            f"| {item['id']} | {item['type']} | {item['passed']} | "
            f"{item['expected']} | {item['actual']} | {detail} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ResearchAgent eval suites")
    parser.add_argument("--suite", choices=["security"], default="security")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    if args.suite == "security":
        report = run_security_suite()
    else:
        raise ValueError(f"Unsupported suite: {args.suite}")

    print(json.dumps({
        "suite": report["suite"],
        "total": report["total"],
        "passed": report["passed"],
        "accuracy": report["accuracy"],
    }, ensure_ascii=False))

    if not args.no_report:
        path = write_report(report)
        print(f"report: {os.path.relpath(path, ROOT)}")

    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
