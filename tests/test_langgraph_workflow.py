import unittest

from agent.langgraph_workflow import (
    LangGraphWorkflowBuilder,
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


class LangGraphWorkflowBuilderTests(unittest.TestCase):
    def test_describe_exposes_runtime_graph_shape_without_dependency(self):
        spec = LangGraphWorkflowBuilder(importer=missing_importer).describe()

        self.assertEqual(spec["entry_node"], "classify")
        self.assertEqual(spec["finish_node"], "finish")
        self.assertIn("retrieve_memory", spec["nodes"])
        self.assertIn("route_classified", spec["conditional_routes"])
        self.assertIn("route_intent", spec["conditional_routes"])
        self.assertIn("reflect", spec["conditional_routes"])
        self.assertIn({"from": "plan", "to": "validate_plan"}, spec["edges"])

    def test_availability_reflects_optional_dependency(self):
        self.assertFalse(LangGraphWorkflowBuilder(importer=missing_importer).is_available())
        self.assertTrue(LangGraphWorkflowBuilder(importer=fake_importer).is_available())

    def test_build_raises_clear_error_when_langgraph_missing(self):
        workflow = LangGraphWorkflowBuilder(importer=missing_importer)
        with self.assertRaises(LangGraphUnavailableError):
            workflow.build()

    def test_build_creates_langgraph_builder_with_routes(self):
        graph = LangGraphWorkflowBuilder(importer=fake_importer).build()
        builder = graph.builder

        self.assertIn("classify", builder.nodes)
        self.assertEqual(builder.state_type.__name__, "ResearchWorkflowState")
        self.assertIn(("__start__", "classify"), builder.edges)
        self.assertIn(("finish", "__end__"), builder.edges)
        self.assertEqual(len(builder.conditional_edges), 3)

        route_names = {start for start, _, _ in builder.conditional_edges}
        self.assertEqual(route_names, {"route_classified", "route_intent", "reflect"})

    def test_build_supports_custom_handlers_and_checkpointer(self):
        marker = object()

        def classify_handler(state):
            return {"intent": "direct_search"}

        graph = LangGraphWorkflowBuilder(importer=fake_importer).build(
            handlers={"classify": classify_handler},
            checkpointer=marker,
        )

        self.assertIs(graph.builder.nodes["classify"], classify_handler)
        self.assertIs(graph.checkpointer, marker)

    def test_build_accepts_runtime_specific_state_schema(self):
        class CustomState(dict):
            pass

        graph = LangGraphWorkflowBuilder(
            importer=fake_importer,
            state_schema=CustomState,
        ).build()

        self.assertIs(graph.builder.state_type, CustomState)

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
