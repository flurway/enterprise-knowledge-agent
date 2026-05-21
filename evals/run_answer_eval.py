#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.answer_eval import run_answer_eval
from evals.metrics import format_percent


def write_report(report: dict) -> Path:
    report_dir = ROOT / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"answer_validation_{stamp}.md"
    lines = [
        "# Eval Report: answer_validation",
        "",
        f"- Total: {report['total']}",
        f"- Passed: {report['passed']}",
        f"- Accuracy: {format_percent(report['accuracy'])}",
        "",
        "| ID | Passed | Expected OK | Actual OK | Unknown Citations | Missing Citations | Unsafe Signals |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        validation = item["validation"]
        lines.append(
            f"| {item['id']} | {item['passed']} | {item['expected_ok']} | {item['actual_ok']} | "
            f"{validation['unknown_citation_ids']} | {validation['missing_citations']} | "
            f"{validation['unsafe_compliance_signals']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run answer-level validation eval")
    parser.add_argument("--cases", default="evals/cases/answer_validation_cases.jsonl")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    report = run_answer_eval(ROOT / args.cases)
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

