import unittest

from agent.policies import ToolRisk
from agent.tool_registry import (
    DEFAULT_TOOL_REGISTRY,
    ToolBackend,
    ToolRegistry,
    ToolScope,
    ToolSpec,
    policy_registry_mismatches,
)


class ToolRegistryTest(unittest.TestCase):
    def test_default_registry_matches_policy_names(self):
        self.assertEqual(policy_registry_mismatches(), [])

    def test_registry_exports_mcp_manifest(self):
        manifest = DEFAULT_TOOL_REGISTRY.to_mcp_manifest()
        tools = {tool["name"]: tool for tool in manifest["tools"]}

        self.assertEqual(manifest["version"], "0.1")
        self.assertIn("search_knowledge_base", tools)
        self.assertIn("code_search", tools)
        self.assertEqual(tools["code_search"]["backend"], "mcp")
        self.assertEqual(tools["code_search"]["mcp_server"], "code")
        self.assertIn("read", tools["code_search"]["scopes"])

    def test_read_only_classification_blocks_write_execute_memory(self):
        read_tool = DEFAULT_TOOL_REGISTRY.get("web_search")
        write_tool = DEFAULT_TOOL_REGISTRY.get("code_patch")
        execute_tool = DEFAULT_TOOL_REGISTRY.get("run_tests")
        memory_tool = DEFAULT_TOOL_REGISTRY.get("memory_write")

        self.assertTrue(read_tool.is_read_only())
        self.assertFalse(write_tool.is_read_only())
        self.assertFalse(execute_tool.is_read_only())
        self.assertFalse(memory_tool.is_read_only())

    def test_disabled_tool_fails_validation(self):
        registry = ToolRegistry([
            ToolSpec(
                name="experimental_tool",
                description="Disabled test tool",
                scopes=(ToolScope.READ,),
                risk=ToolRisk.LOW,
                backend=ToolBackend.MCP,
                enabled=False,
            )
        ])

        ok, reason = registry.validate_action("experimental_tool")

        self.assertFalse(ok)
        self.assertIn("disabled", reason)


if __name__ == "__main__":
    unittest.main()
