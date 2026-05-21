"""
Optional LangGraph adapter for the reliable agent runtime.

This module does not replace AgentRuntime. It exposes the current runtime
control-flow as a LangGraph-compatible graph when langgraph is installed, while
keeping the project runnable without that optional dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, TypedDict

from agent.runtime import RuntimeNode


class LangGraphUnavailableError(ImportError):
    """Raised when the optional langgraph dependency is required but missing."""


class ReliableAgentGraphState(TypedDict, total=False):
    run_id: str
    session_id: str
    user_message: str
    refined_query: str
    intent: str
    plan: dict[str, Any]
    result: dict[str, Any]
    needs_replan: bool
    error: str
    visited_nodes: list[str]


GraphNodeHandler = Callable[[ReliableAgentGraphState], dict[str, Any] | Awaitable[dict[str, Any]]]
LangGraphImporter = Callable[[], tuple[Any, str, str]]


DEFAULT_LANGGRAPH_NODES: tuple[str, ...] = (
    RuntimeNode.CLASSIFY.value,
    "route_classified",
    RuntimeNode.RETRIEVE_MEMORY.value,
    "route_intent",
    RuntimeNode.DIRECT_SEARCH.value,
    RuntimeNode.PLAN.value,
    RuntimeNode.VALIDATE_PLAN.value,
    RuntimeNode.EXECUTE_STEP.value,
    RuntimeNode.REFLECT.value,
    RuntimeNode.REPLAN.value,
    RuntimeNode.SYNTHESIZE.value,
    RuntimeNode.VALIDATE_ANSWER.value,
    RuntimeNode.SAVE_MEMORY.value,
    RuntimeNode.FINISH.value,
)


DEFAULT_LANGGRAPH_EDGES: tuple[tuple[str, str], ...] = (
    (RuntimeNode.CLASSIFY.value, "route_classified"),
    (RuntimeNode.RETRIEVE_MEMORY.value, "route_intent"),
    (RuntimeNode.DIRECT_SEARCH.value, RuntimeNode.VALIDATE_ANSWER.value),
    (RuntimeNode.PLAN.value, RuntimeNode.VALIDATE_PLAN.value),
    (RuntimeNode.VALIDATE_PLAN.value, RuntimeNode.EXECUTE_STEP.value),
    (RuntimeNode.EXECUTE_STEP.value, RuntimeNode.REFLECT.value),
    (RuntimeNode.REPLAN.value, RuntimeNode.VALIDATE_PLAN.value),
    (RuntimeNode.SYNTHESIZE.value, RuntimeNode.VALIDATE_ANSWER.value),
    (RuntimeNode.VALIDATE_ANSWER.value, RuntimeNode.SAVE_MEMORY.value),
    (RuntimeNode.SAVE_MEMORY.value, RuntimeNode.FINISH.value),
)


def _import_langgraph() -> tuple[Any, str, str]:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise LangGraphUnavailableError(
            "LangGraph is optional. Install it with `pip install langgraph` "
            "to build a compiled graph."
        ) from exc
    return StateGraph, START, END


def route_after_classify(state: ReliableAgentGraphState) -> str:
    """Route terminal chitchat/clarification/conflict answers to finish."""
    return RuntimeNode.FINISH.value if state.get("short_circuit") else RuntimeNode.RETRIEVE_MEMORY.value


def route_after_intent(state: ReliableAgentGraphState) -> str:
    """Route direct search separately from deeper plan/execute workflows."""
    return (
        RuntimeNode.DIRECT_SEARCH.value
        if state.get("intent") == "direct_search"
        else RuntimeNode.PLAN.value
    )


def route_after_reflection(state: ReliableAgentGraphState) -> str:
    """Route failed/insufficient execution back to replan when requested."""
    if state.get("skip_synthesis"):
        return RuntimeNode.VALIDATE_ANSWER.value
    return RuntimeNode.REPLAN.value if state.get("needs_replan") else RuntimeNode.SYNTHESIZE.value


def _passthrough_node(node_name: str) -> GraphNodeHandler:
    def run(state: ReliableAgentGraphState) -> dict[str, Any]:
        visited = list(state.get("visited_nodes", []))
        visited.append(node_name)
        return {"visited_nodes": visited}

    run.__name__ = f"{node_name}_node"
    return run


@dataclass
class LangGraphAdapter:
    """
    Build a LangGraph view of the existing runtime nodes.

    Node handlers are injectable so production code can bind each LangGraph node
    to the existing orchestrator/runtime functions incrementally. Missing
    handlers become traceable passthrough nodes, which is useful for graph
    visualization and early integration tests.
    """

    importer: LangGraphImporter = _import_langgraph
    nodes: tuple[str, ...] = DEFAULT_LANGGRAPH_NODES
    edges: tuple[tuple[str, str], ...] = DEFAULT_LANGGRAPH_EDGES
    conditional_routes: dict[str, Callable[[ReliableAgentGraphState], str]] = field(default_factory=lambda: {
        "route_classified": route_after_classify,
        "route_intent": route_after_intent,
        RuntimeNode.REFLECT.value: route_after_reflection,
    })

    def describe(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [{"from": start, "to": end} for start, end in self.edges],
            "conditional_routes": sorted(self.conditional_routes),
            "entry_node": RuntimeNode.CLASSIFY.value,
            "finish_node": RuntimeNode.FINISH.value,
        }

    def is_available(self) -> bool:
        try:
            self.importer()
            return True
        except LangGraphUnavailableError:
            return False

    def build(
        self,
        handlers: Mapping[str, GraphNodeHandler] | None = None,
        *,
        checkpointer: Any | None = None,
        compile_graph: bool = True,
    ) -> Any:
        StateGraph, START, END = self.importer()
        builder = StateGraph(ReliableAgentGraphState)
        handler_map = dict(handlers or {})

        for node in self.nodes:
            builder.add_node(node, handler_map.get(node, _passthrough_node(node)))

        builder.add_edge(START, RuntimeNode.CLASSIFY.value)
        for start, end in self.edges:
            builder.add_edge(start, end)

        builder.add_conditional_edges(
            "route_classified",
            self.conditional_routes["route_classified"],
            {
                RuntimeNode.RETRIEVE_MEMORY.value: RuntimeNode.RETRIEVE_MEMORY.value,
                RuntimeNode.FINISH.value: RuntimeNode.FINISH.value,
            },
        )
        builder.add_conditional_edges(
            "route_intent",
            self.conditional_routes["route_intent"],
            {
                RuntimeNode.DIRECT_SEARCH.value: RuntimeNode.DIRECT_SEARCH.value,
                RuntimeNode.PLAN.value: RuntimeNode.PLAN.value,
            },
        )
        builder.add_conditional_edges(
            RuntimeNode.REFLECT.value,
            self.conditional_routes[RuntimeNode.REFLECT.value],
            {
                RuntimeNode.REPLAN.value: RuntimeNode.REPLAN.value,
                RuntimeNode.SYNTHESIZE.value: RuntimeNode.SYNTHESIZE.value,
                RuntimeNode.VALIDATE_ANSWER.value: RuntimeNode.VALIDATE_ANSWER.value,
            },
        )
        builder.add_edge(RuntimeNode.FINISH.value, END)

        if not compile_graph:
            return builder
        if checkpointer is None:
            return builder.compile()
        return builder.compile(checkpointer=checkpointer)
