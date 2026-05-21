from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from agent.langgraph.state import ResearchGraphState
from agent.runtime import RuntimeNode


class LangGraphUnavailableError(ImportError):
    """Raised when the langgraph runtime dependency is missing."""


GraphNodeHandler = Callable[[ResearchGraphState], dict[str, Any] | Awaitable[dict[str, Any]]]
LangGraphImporter = Callable[[], tuple[Any, str, str]]


GRAPH_NODES: tuple[str, ...] = (
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


GRAPH_EDGES: tuple[tuple[str, str], ...] = (
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


def import_langgraph() -> tuple[Any, str, str]:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise LangGraphUnavailableError(
            "LangGraph is required for the langgraph runtime. Install it with `pip install langgraph` "
            "to build a compiled graph."
        ) from exc
    return StateGraph, START, END


def route_after_classify(state: ResearchGraphState) -> str:
    return RuntimeNode.FINISH.value if state.get("short_circuit") else RuntimeNode.RETRIEVE_MEMORY.value


def route_after_intent(state: ResearchGraphState) -> str:
    return RuntimeNode.DIRECT_SEARCH.value if state.get("intent") == "direct_search" else RuntimeNode.PLAN.value


def route_after_reflection(state: ResearchGraphState) -> str:
    if state.get("skip_synthesis"):
        return RuntimeNode.VALIDATE_ANSWER.value
    return RuntimeNode.REPLAN.value if state.get("needs_replan") else RuntimeNode.SYNTHESIZE.value


@dataclass
class ResearchGraphBuilder:
    importer: LangGraphImporter = import_langgraph
    state_schema: Any = ResearchGraphState
    nodes: tuple[str, ...] = GRAPH_NODES
    edges: tuple[tuple[str, str], ...] = GRAPH_EDGES
    conditional_routes: dict[str, Callable[[ResearchGraphState], str]] = field(default_factory=lambda: {
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
        handlers: Mapping[str, GraphNodeHandler],
        *,
        checkpointer: Any | None = None,
        compile_graph: bool = True,
    ) -> Any:
        StateGraph, START, END = self.importer()
        builder = StateGraph(self.state_schema)

        for node in self.nodes:
            if node not in handlers:
                raise ValueError(f"Missing LangGraph node handler: {node}")
            builder.add_node(node, handlers[node])

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

