#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.memory_write_eval import run_memory_write_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run memory write gate eval")
    parser.add_argument("--cases", default=str(ROOT / "evals" / "cases" / "memory_write_cases.jsonl"))
    parser.add_argument("--no-report", action="store_true", help="Accepted for consistency; this runner prints JSON only")
    args = parser.parse_args()

    report = run_memory_write_eval(Path(args.cases))
    print(json.dumps({
        "suite": report["suite"],
        "total": report["total"],
        "passed": report["passed"],
        "accuracy": report["accuracy"],
    }, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
