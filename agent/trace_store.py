"""
SQLite-backed Agent trace persistence.

The store keeps both normalized tables for common queries and the full JSON
trace for replay/debug. This gives the project a concrete persistence layer
without committing to a production database too early.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from agent.state import AgentRun


SCHEMA_VERSION = 1


class AgentTraceStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    intent TEXT,
                    refined_query TEXT,
                    status TEXT NOT NULL,
                    started_at_ms INTEGER NOT NULL,
                    finished_at_ms INTEGER,
                    tool_call_count INTEGER NOT NULL DEFAULT 0,
                    event_count INTEGER NOT NULL DEFAULT 0,
                    trace_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_steps (
                    run_id TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    depends_on_json TEXT NOT NULL,
                    expected_output TEXT,
                    verification TEXT,
                    failure_type TEXT,
                    error TEXT,
                    started_at_ms INTEGER,
                    finished_at_ms INTEGER,
                    PRIMARY KEY (run_id, step_id),
                    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    risk_level TEXT,
                    policy_decision TEXT,
                    latency_ms INTEGER NOT NULL,
                    error TEXT,
                    input_preview TEXT,
                    output_preview TEXT,
                    timestamp_ms INTEGER NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_session_time ON agent_runs(session_id, started_at_ms)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_run_step ON tool_calls(run_id, step_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_events_run_time ON trace_events(run_id, timestamp_ms)")
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def save_run(self, run: AgentRun) -> None:
        trace = run.to_dict()
        with self._connect() as conn:
            conn.execute("DELETE FROM plan_steps WHERE run_id = ?", (run.run_id,))
            conn.execute("DELETE FROM tool_calls WHERE run_id = ?", (run.run_id,))
            conn.execute("DELETE FROM trace_events WHERE run_id = ?", (run.run_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_runs (
                    run_id, session_id, user_message, intent, refined_query, status,
                    started_at_ms, finished_at_ms, tool_call_count, event_count, trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.session_id,
                    run.user_message,
                    run.intent,
                    run.refined_query,
                    run.status.value if hasattr(run.status, "value") else str(run.status),
                    run.started_at_ms,
                    run.finished_at_ms,
                    len(run.tool_calls),
                    len(run.events),
                    json.dumps(trace, ensure_ascii=False),
                ),
            )

            for step in run.steps:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO plan_steps (
                        run_id, step_id, action, description, status, depends_on_json,
                        expected_output, verification, failure_type, error,
                        started_at_ms, finished_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        step.step_id,
                        step.action,
                        step.description,
                        step.status.value if hasattr(step.status, "value") else str(step.status),
                        json.dumps(step.depends_on, ensure_ascii=False),
                        step.expected_output,
                        step.verification,
                        step.failure_type.value if hasattr(step.failure_type, "value") else str(step.failure_type),
                        step.error,
                        step.started_at_ms,
                        step.finished_at_ms,
                    ),
                )

            for call in run.tool_calls:
                conn.execute(
                    """
                    INSERT INTO tool_calls (
                        run_id, step_id, tool_name, success, risk_level, policy_decision,
                        latency_ms, error, input_preview, output_preview, timestamp_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        call.step_id,
                        call.tool_name,
                        1 if call.success else 0,
                        call.risk_level,
                        call.policy_decision,
                        call.latency_ms,
                        call.error,
                        call.input_preview,
                        call.output_preview,
                        call.timestamp_ms,
                    ),
                )

            for event in run.events:
                conn.execute(
                    """
                    INSERT INTO trace_events (
                        run_id, event_type, message, metadata_json, timestamp_ms
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        event.event_type,
                        event.message,
                        json.dumps(event.metadata, ensure_ascii=False),
                        event.timestamp_ms,
                    ),
                )

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT trace_json FROM agent_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            return json.loads(row["trace_json"])

    def list_runs(self, limit: int = 20, session_id: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT run_id, session_id, intent, status, started_at_ms, finished_at_ms,
                           tool_call_count, event_count
                    FROM agent_runs
                    WHERE session_id = ?
                    ORDER BY started_at_ms DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT run_id, session_id, intent, status, started_at_ms, finished_at_ms,
                           tool_call_count, event_count
                    FROM agent_runs
                    ORDER BY started_at_ms DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

