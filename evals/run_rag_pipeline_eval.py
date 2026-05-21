#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.metrics import format_percent
from evals.rag_pipeline_eval import report_summary, run_rag_pipeline_eval


def write_report(report: dict) -> Path:
    report_dir = ROOT / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"rag_pipeline_{stamp}.md"
    lines = [
        "# Eval Report: rag_pipeline",
        "",
        f"- Index Backend: {report['index_backend']}",
        f"- Total: {report['total']}",
        f"- Top-1 Accuracy: {format_percent(report['top1_accuracy'])}",
        f"- Top-k Recall: {format_percent(report['topk_recall'])}",
        f"- Must-Include Accuracy: {format_percent(report['must_include_accuracy'])}",
        f"- Citation Start Accuracy: {format_percent(report['citation_start_accuracy'])}",
        "",
        "| ID | Top-1 | Top-k | Expected | Top Doc | Rank | Citation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['id']} | {item['top1_pass']} | {item['topk_pass']} | "
            f"{item['expected_doc']} | {item['top_doc']} | {item['rank']} | {item['citation_start_pass']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Run fixture KB through the RAG retrieval pipeline")
    parser.add_argument("--kb-dir", default="evals/fixtures/knowledge_base")
    parser.add_argument("--cases", default="evals/cases/kb_retrieval_cases.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    report = await run_rag_pipeline_eval(
        kb_dir=ROOT / args.kb_dir,
        cases_path=ROOT / args.cases,
        top_k=args.top_k,
    )
    print(json.dumps(report_summary(report), ensure_ascii=False))

    if not args.no_report:
        path = write_report(report)
        print(f"report: {os.path.relpath(path, ROOT)}")

    passed = (
        report["top1_accuracy"] == 1.0
        and report["topk_recall"] == 1.0
        and report["must_include_accuracy"] == 1.0
        and report["citation_start_accuracy"] == 1.0
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
