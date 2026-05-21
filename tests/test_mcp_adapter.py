import asyncio
import unittest

from agent.mcp_adapter import McpServerConfig, McpToolAdapter, McpToolUnavailableError
from agent.tool_dispatcher import ToolDispatcher, ToolInvocationContext


class McpToolAdapterTest(unittest.TestCase):
    def test_registers_mcp_tools_with_dispatcher(self):
        dispatcher = ToolDispatcher()
        adapter = McpToolAdapter()

        registered = adapter.register_with(dispatcher)

        self.assertIn("code_search", registered)
        self.assertIn("run_tests", registered)
        self.assertIn("code_patch", registered)
        self.assertTrue(dispatcher.has_handler("code_search"))

    def test_unconfigured_mcp_server_returns_structured_error(self):
        async def run():
            dispatcher = ToolDispatcher()
            McpToolAdapter().register_with(dispatcher)
            await dispatcher.dispatch(
                "code_search",
                ToolInvocationContext(action="code_search", params={"query": "runtime"}),
            )

        with self.assertRaises(McpToolUnavailableError) as cm:
            asyncio.run(run())

        self.assertEqual(cm.exception.code, "mcp_server_unconfigured")
        self.assertEqual(cm.exception.tool_name, "code_search")
        self.assertEqual(cm.exception.backend, "mcp")

    def test_disabled_mcp_server_is_distinguishable(self):
        async def run():
            dispatcher = ToolDispatcher()
            adapter = McpToolAdapter(servers={"code": McpServerConfig(name="code", enabled=False)})
            adapter.register_with(dispatcher)
            await dispatcher.dispatch(
                "code_search",
                ToolInvocationContext(action="code_search", params={"query": "runtime"}),
            )

        with self.assertRaises(McpToolUnavailableError) as cm:
            asyncio.run(run())

        self.assertEqual(cm.exception.code, "mcp_server_disabled")

    def test_manifest_reports_server_status(self):
        adapter = McpToolAdapter(servers={"code": McpServerConfig(name="code", enabled=False)})
        manifest = adapter.manifest()
        tools = {tool["name"]: tool for tool in manifest["tools"]}

        self.assertIn("code", manifest["servers"])
        self.assertTrue(tools["code_search"]["server_configured"])
        self.assertFalse(tools["code_search"]["server_enabled"])


if __name__ == "__main__":
    unittest.main()
