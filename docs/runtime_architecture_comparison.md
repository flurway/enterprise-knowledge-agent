# Runtime Architecture Comparison

This project keeps two runtime shapes intentionally comparable.

## Custom Reliable Runtime

The custom runtime is the baseline implementation.

- Control flow: `AgentRuntime` + todo-aware `ReActLoop`
- State: `AgentRun`, `PlanStepState`, `TodoItem`, `TraceEvent`
- Tool execution: `ToolRegistry`, `ToolDispatcher`, `ToolPolicy`, hooks
- Strengths: low dependency, explicit learning value, easy unit testing
- Limitation: durable execution, resume, interrupt, and human-in-the-loop would need more custom infrastructure

## LangGraph Runtime

The LangGraph runtime moves high-level orchestration into a graph while reusing
the existing reliability modules.

- Control flow: `StateGraph` nodes and conditional edges
- State: `ResearchGraphState`
- Graph definition: `agent/langgraph_adapter.py`
- Runtime binding: `agent/langgraph_runtime.py`
- Strengths: graph-native control flow, clearer branch structure, future path to checkpoint/resume/human confirmation
- Tradeoff: more framework coupling and a dependency on LangGraph

## Preserved Components

Both runtime backends reuse:

- RAG retrieval and citation tracing
- Tool registry and tool dispatcher
- Tool permission policy and hooks
- Prompt-injection detection and untrusted-context propagation
- Memory write gate, memory conflict handling, and retrieval budget
- Answer validation and offline evals

## How To Compare

Run the custom runtime:

```bash
AGENT_RUNTIME_BACKEND=custom python cli.py chat
```

Run the LangGraph runtime:

```bash
AGENT_RUNTIME_BACKEND=langgraph python cli.py chat
```

Compare branches if the migration is developed separately:

```bash
git diff master..codex/langgraph-runtime --stat
git diff master..codex/langgraph-runtime -- agent/orchestrator.py agent/langgraph_runtime.py
```

