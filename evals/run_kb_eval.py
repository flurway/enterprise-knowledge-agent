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

from evals.kb_eval import run_kb_eval
from evals.metrics import format_percent


def write_report(report: dict) -> Path:
    report_dir = ROOT / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"kb_retrieval_{stamp}.md"
    lines = [
        "# Eval Report: kb_retrieval",
        "",
        f"- Total: {report['total']}",
        f"- Top-1 Accuracy: {format_percent(report['top1_accuracy'])}",
        f"- Top-k Recall: {format_percent(report['topk_recall'])}",
        f"- Must-Include Accuracy: {format_percent(report['must_include_accuracy'])}",
        "",
        "| ID | Top-1 | Top-k | Expected | Top Doc | Rank |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['id']} | {item['top1_pass']} | {item['topk_pass']} | "
            f"{item['expected_doc']} | {item['top_doc']} | {item['rank']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline knowledge-base retrieval eval")
    parser.add_argument("--kb-dir", default="evals/fixtures/knowledge_base")
    parser.add_argument("--cases", default="evals/cases/kb_retrieval_cases.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    report = run_kb_eval(
        kb_dir=ROOT / args.kb_dir,
        cases_path=ROOT / args.cases,
        top_k=args.top_k,
    )
    print(json.dumps({
        "suite": report["suite"],
        "total": report["total"],
        "top1_accuracy": report["top1_accuracy"],
        "topk_recall": report["topk_recall"],
        "must_include_accuracy": report["must_include_accuracy"],
    }, ensure_ascii=False))

    if not args.no_report:
        path = write_report(report)
        print(f"report: {os.path.relpath(path, ROOT)}")

    return 0 if report["top1_accuracy"] == 1.0 and report["must_include_accuracy"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

