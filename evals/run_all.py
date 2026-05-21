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

from evals.answer_eval import run_answer_eval
from evals.kb_eval import run_kb_eval
from evals.memory_conflict_eval import run_memory_conflict_eval
from evals.memory_retrieval_eval import run_memory_retrieval_eval
from evals.memory_write_eval import run_memory_write_eval
from evals.metrics import format_percent
from evals.rag_pipeline_eval import report_summary, run_rag_pipeline_eval
from evals.run_evals import run_security_suite


def suite_passed(name: str, report: dict) -> bool:
    if name == "security":
        return report["passed"] == report["total"]
    if name == "kb_retrieval":
        return (
            report["top1_accuracy"] == 1.0
            and report["topk_recall"] == 1.0
            and report["must_include_accuracy"] == 1.0
        )
    if name == "rag_pipeline":
        return (
            report["top1_accuracy"] == 1.0
            and report["topk_recall"] == 1.0
            and report["must_include_accuracy"] == 1.0
            and report["citation_start_accuracy"] == 1.0
        )
    if name == "answer_validation":
        return report["passed"] == report["total"]
    if name == "memory_write":
        return report["passed"] == report["total"]
    if name == "memory_conflict":
        return report["passed"] == report["total"]
    if name == "memory_retrieval":
        return report["passed"] == report["total"]
    return False


def summarize_report(name: str, report: dict) -> dict:
    if name == "security":
        return {
            "suite": name,
            "passed": suite_passed(name, report),
            "total": report["total"],
            "accuracy": report["accuracy"],
        }
    if name == "kb_retrieval":
        return {
            "suite": name,
            "passed": suite_passed(name, report),
            "total": report["total"],
            "top1_accuracy": report["top1_accuracy"],
            "topk_recall": report["topk_recall"],
            "must_include_accuracy": report["must_include_accuracy"],
        }
    if name == "rag_pipeline":
        summary = report_summary(report)
        summary["passed"] = suite_passed(name, report)
        return summary
    if name == "answer_validation":
        return {
            "suite": name,
            "passed": suite_passed(name, report),
            "total": report["total"],
            "accuracy": report["accuracy"],
        }
    if name == "memory_write":
        return {
            "suite": name,
            "passed": suite_passed(name, report),
            "total": report["total"],
            "accuracy": report["accuracy"],
        }
    if name == "memory_conflict":
        return {
            "suite": name,
            "passed": suite_passed(name, report),
            "total": report["total"],
            "accuracy": report["accuracy"],
        }
    if name == "memory_retrieval":
        return {
            "suite": name,
            "passed": suite_passed(name, report),
            "total": report["total"],
            "accuracy": report["accuracy"],
        }
    return {"suite": name, "passed": False}


def write_report(summary: dict) -> Path:
    report_dir = ROOT / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"all_{stamp}.md"
    lines = [
        "# Eval Report: all",
        "",
        f"- Overall Passed: {summary['overall_passed']}",
        "",
        "| Suite | Passed | Key Metrics |",
        "| --- | --- | --- |",
    ]
    for suite in summary["suites"]:
        metrics = []
        for key, value in suite.items():
            if key in {"suite", "passed"}:
                continue
            if isinstance(value, float):
                metrics.append(f"{key}={format_percent(value)}")
            else:
                metrics.append(f"{key}={value}")
        lines.append(f"| {suite['suite']} | {suite['passed']} | {'; '.join(metrics)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def run_all(no_report: bool = False) -> dict:
    security = run_security_suite()
    kb = run_kb_eval(
        kb_dir=ROOT / "evals" / "fixtures" / "knowledge_base",
        cases_path=ROOT / "evals" / "cases" / "kb_retrieval_cases.jsonl",
    )
    rag = await run_rag_pipeline_eval(
        kb_dir=ROOT / "evals" / "fixtures" / "knowledge_base",
        cases_path=ROOT / "evals" / "cases" / "kb_retrieval_cases.jsonl",
    )
    answer = run_answer_eval(ROOT / "evals" / "cases" / "answer_validation_cases.jsonl")
    memory_write = run_memory_write_eval(ROOT / "evals" / "cases" / "memory_write_cases.jsonl")
    memory_conflict = run_memory_conflict_eval(ROOT / "evals" / "cases" / "memory_conflict_cases.jsonl")
    memory_retrieval = run_memory_retrieval_eval(ROOT / "evals" / "cases" / "memory_retrieval_cases.jsonl")

    suites = [
        summarize_report("security", security),
        summarize_report("kb_retrieval", kb),
        summarize_report("rag_pipeline", rag),
        summarize_report("answer_validation", answer),
        summarize_report("memory_write", memory_write),
        summarize_report("memory_conflict", memory_conflict),
        summarize_report("memory_retrieval", memory_retrieval),
    ]
    summary = {
        "overall_passed": all(suite["passed"] for suite in suites),
        "suites": suites,
    }
    if not no_report:
        path = write_report(summary)
        summary["report_path"] = os.path.relpath(path, ROOT)
    return summary


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Run all offline ResearchAgent eval suites")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    summary = await run_all(no_report=args.no_report)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
