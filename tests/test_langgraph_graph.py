import unittest

from agent.langgraph.graph import (
    ResearchGraphBuilder,
    LangGraphUnavailableError,
    route_after_classify,
    route_after_intent,
    route_after_reflection,
)


class FakeCompiledGraph:
    def __init__(self, builder, checkpointer=None):
        self.builder = builder
        self.checkpointer = checkpointer


class FakeStateGraph:
    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []
        self.compile_calls = []

    def add_node(self, name, handler):
        self.nodes[name] = handler

    def add_edge(self, start, end):
        self.edges.append((start, end))

    def add_conditional_edges(self, start, route_fn, path_map):
        self.conditional_edges.append((start, route_fn, path_map))

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)
        return FakeCompiledGraph(self, kwargs.get("checkpointer"))


def fake_importer():
    return FakeStateGraph, "__start__", "__end__"


def missing_importer():
    raise LangGraphUnavailableError("missing")


def all_handlers():
    return {node: (lambda state: {}) for node in ResearchGraphBuilder().nodes}


class ResearchGraphBuilderTests(unittest.TestCase):
    def test_describe_exposes_runtime_graph_shape_without_dependency(self):
        spec = ResearchGraphBuilder(importer=missing_importer).describe()

        self.assertEqual(spec["entry_node"], "classify")
        self.assertEqual(spec["finish_node"], "finish")
        self.assertIn("retrieve_memory", spec["nodes"])
        self.assertIn("route_classified", spec["conditional_routes"])
        self.assertIn("route_intent", spec["conditional_routes"])
        self.assertIn("reflect", spec["conditional_routes"])
        self.assertIn({"from": "plan", "to": "validate_plan"}, spec["edges"])

    def test_availability_reflects_optional_dependency(self):
        self.assertFalse(ResearchGraphBuilder(importer=missing_importer).is_available())
        self.assertTrue(ResearchGraphBuilder(importer=fake_importer).is_available())

    def test_build_raises_clear_error_when_langgraph_missing(self):
        graph_builder = ResearchGraphBuilder(importer=missing_importer)
        with self.assertRaises(LangGraphUnavailableError):
            graph_builder.build(handlers={})

    def test_build_creates_langgraph_builder_with_routes(self):
        graph = ResearchGraphBuilder(importer=fake_importer).build(handlers=all_handlers())
        builder = graph.builder

        self.assertIn("classify", builder.nodes)
        self.assertEqual(builder.state_type.__name__, "ResearchGraphState")
        self.assertIn(("__start__", "classify"), builder.edges)
        self.assertIn(("finish", "__end__"), builder.edges)
        self.assertEqual(len(builder.conditional_edges), 3)

        route_names = {start for start, _, _ in builder.conditional_edges}
        self.assertEqual(route_names, {"route_classified", "route_intent", "reflect"})

    def test_build_supports_custom_handlers_and_checkpointer(self):
        marker = object()

        def classify_handler(state):
            return {"intent": "direct_search"}

        handlers = all_handlers()
        handlers["classify"] = classify_handler
        graph = ResearchGraphBuilder(importer=fake_importer).build(
            handlers=handlers,
            checkpointer=marker,
        )

        self.assertIs(graph.builder.nodes["classify"], classify_handler)
        self.assertIs(graph.checkpointer, marker)

    def test_build_accepts_runtime_specific_state_schema(self):
        class CustomState(dict):
            pass

        graph = ResearchGraphBuilder(
            importer=fake_importer,
            state_schema=CustomState,
        ).build(handlers=all_handlers())

        self.assertIs(graph.builder.state_type, CustomState)

    def test_build_requires_explicit_handlers_for_all_nodes(self):
        with self.assertRaises(ValueError) as cm:
            ResearchGraphBuilder(importer=fake_importer).build(handlers={"classify": lambda state: {}})

        self.assertIn("Missing LangGraph node handler", str(cm.exception))

    def test_routes_match_runtime_decisions(self):
        self.assertEqual(route_after_classify({"short_circuit": True}), "finish")
        self.assertEqual(route_after_classify({"short_circuit": False}), "retrieve_memory")
        self.assertEqual(route_after_intent({"intent": "direct_search"}), "direct_search")
        self.assertEqual(route_after_intent({"intent": "deep_research"}), "plan")
        self.assertEqual(route_after_reflection({"needs_replan": True}), "replan")
        self.assertEqual(route_after_reflection({"needs_replan": False}), "synthesize")
        self.assertEqual(route_after_reflection({"skip_synthesis": True}), "validate_answer")


if __name__ == "__main__":
    unittest.main()
