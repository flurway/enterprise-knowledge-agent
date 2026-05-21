"""
Helpers for propagating untrusted-context risk across tool steps.

RAG/Web outputs are evidence, not instructions. If a previous tool result was
flagged as suspicious, later steps that depend on it should be treated as
triggered by untrusted context unless an explicit policy allowlist says
otherwise.
"""
from __future__ import annotations

from typing import Any, Mapping


def suspicious_dependency_ids(execution_context: Mapping[int, Any], depends_on: list[int]) -> list[int]:
    suspicious: list[int] = []
    for dep_id in depends_on:
        result = execution_context.get(dep_id)
        if result is None:
            continue
        data = getattr(result, "data", {}) or {}
        injection_scan = data.get("injection_scan", {}) or {}
        if injection_scan.get("is_suspicious") or data.get("triggered_by_untrusted_context"):
            suspicious.append(dep_id)
    return suspicious
