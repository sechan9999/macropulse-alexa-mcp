"""
tests/test_mcp_tool_list.py
─────────────────────────────────────────────────────────────────
/health and the landing page list exactly the tools the MCP server registers. /health used to carry a
hand-written list that missed get_morning_brief and get_equity_report. Offline; no market data.
"""
import asyncio
import unittest

from starlette.testclient import TestClient

from src import alexa_mcp_server as srv


class TestToolLists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered = {t.name for t in asyncio.run(srv.create_mcp_server().list_tools())}

    def test_server_registers_nine_tools(self):
        self.assertEqual(len(self.registered), 9)

    def test_health_lists_every_registered_tool(self):
        health = TestClient(srv.build_starlette_app(warm_up=False)).get("/health").json()
        self.assertEqual(sorted(health["available_tools"]), sorted(self.registered))
        self.assertIn("get_equity_report", health["available_tools"])
        self.assertIn("get_morning_brief", health["available_tools"])

    def test_landing_page_lists_every_registered_tool(self):
        self.assertEqual({name for name, _ in srv._LANDING_TOOLS}, self.registered)


if __name__ == "__main__":
    unittest.main()
