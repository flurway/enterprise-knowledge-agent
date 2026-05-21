import asyncio
import unittest

from agent.tool_dispatcher import ToolDispatcher, ToolDispatchError, ToolInvocationContext


class ToolDispatcherTest(unittest.TestCase):
    def test_dispatch_registered_tool(self):
        async def run():
            dispatcher = ToolDispatcher()

            async def handler(context):
                return {
                    "action": context.action,
                    "query": context.params["query"],
                    "dep_context": context.dep_context,
                }

            dispatcher.register("search_knowledge_base", handler)
            return await dispatcher.dispatch(
                "search_knowledge_base",
                ToolInvocationContext(
                    action="search_knowledge_base",
                    params={"query": "memory conflict"},
                    dep_context="previous result",
                ),
            )

        result = asyncio.run(run())
        self.assertEqual(result["action"], "search_knowledge_base")
        self.assertEqual(result["query"], "memory conflict")
        self.assertEqual(result["dep_context"], "previous result")

    def test_register_unknown_tool_is_rejected(self):
        dispatcher = ToolDispatcher()

        async def handler(context):
            return None

        with self.assertRaises(ToolDispatchError):
            dispatcher.register("unknown_tool", handler)

    def test_dispatch_unregistered_enabled_tool_reports_backend(self):
        async def run():
            dispatcher = ToolDispatcher()
            return await dispatcher.dispatch(
                "code_search",
                ToolInvocationContext(action="code_search", params={"query": "AgentRuntime"}),
            )

        with self.assertRaisesRegex(ToolDispatchError, "backend=mcp"):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
