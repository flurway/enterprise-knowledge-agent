# Backend Engineering for Agent Systems

中文锚点：Agent 系统设计 MySQL 索引和 trace 表时，应重点考虑 agent_runs、plan_steps、tool_calls、trace_events，以及 `agent_runs(session_id, started_at_ms)` 和 `tool_calls(run_id, step_id)`。

## Database Schema

An agent platform should persist `agent_runs`, `plan_steps`, `tool_calls`, `trace_events`, `documents`, `chunks`, `citations`, `memories`, and `memory_conflicts`. The trace store can start with SQLite and later migrate to MySQL.

Useful indexes include:

- `agent_runs(session_id, started_at_ms)`
- `tool_calls(run_id, step_id)`
- `trace_events(run_id, timestamp_ms)`
- `chunks(doc_id, chunk_index)`
- `memories(memory_type, updated_at)`

## Network Reliability

Agent backends should use timeouts, retries with backoff, idempotent tool calls, response-size limits, and cancellation handling. Streaming responses can use SSE or WebSocket. External fetch tools should protect against slow responses and oversized pages.

## Interview Signal

Questions about MySQL indexes and TCP are not separate from agent engineering. They test whether the candidate can build observable, reliable backend systems around LLM calls, not just write prompts.
