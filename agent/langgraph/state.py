from __future__ import annotations

from typing import Any, TypedDict


class ResearchGraphState(TypedDict, total=False):
    run_id: str
    session_id: str
    user_message: str
    original_query: str
    refined_query: str
    search_query: str
    intent: str
    intent_result: dict[str, Any]
    sub_questions: list[str]
    memory: Any
    run: Any
    runtime: Any
    plan: dict[str, Any]
    steps: list[dict[str, Any]]
    result: dict[str, Any]
    final_result: dict[str, Any]
    lt_ctx: str
    conflict_notices: list[dict]
    executor: Any
    all_step_results: list[Any]
    failed_result: Any
    failed_step: dict[str, Any]
    reflection: dict[str, Any]
    hallucination_check: dict[str, Any]
    replan_count: int
    plan_validation_errors: list[str]
    needs_replan: bool
    skip_synthesis: bool
    short_circuit: bool
    error: str
