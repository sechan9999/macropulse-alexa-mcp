"""
tests/test_alexa_mcp_transport.py
─────────────────────────────────────────────────────────────────
Real-HTTP tests of the MCP transports served by src.alexa_mcp_server.

The other Alexa tests call the tool functions directly and never touch the
transport, which is how "/mcp always fails with 'Task group is not
initialized'" (MCP app mounted without its lifespan) went unnoticed.
Runs against both mcp 1.x and mcp>=2.
"""
import asyncio
import http.client
import json
import socket
import threading
import time
import unittest
from unittest import mock

import uvicorn

import src.alexa_mcp_server as srv
from src.alexa_mcp_server import _FASTMCP_AVAILABLE, build_starlette_app

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    try:
        from mcp.client.streamable_http import streamable_http_client as http_client  # mcp>=2
    except ImportError:
        from mcp.client.streamable_http import streamablehttp_client as http_client  # mcp 1.x
except ImportError:  # pragma: no cover
    ClientSession = None

EXPECTED_TOOLS = {
    "get_macro_regime", "get_rates_and_spreads", "simulate_portfolio_risk",
    "check_nvda_danger_zone", "scan_quant_signals", "get_expected_returns",
    "simulate_fomc_shock",
}
NOT_MCP_TOOLS = {"ask_macro_analyst"}  # a synchronous LLM call cannot meet the 500 ms budget


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(_FASTMCP_AVAILABLE and ClientSession, "mcp package not installed")
class TestMcpTransports(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.server = uvicorn.Server(uvicorn.Config(
            build_starlette_app(warm_up=False), host="127.0.0.1", port=cls.port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 30
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.05)
        assert cls.server.started, "test server did not start"

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)

    def _http(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    async def _session_roundtrip(self, client_cm):
        async with client_cm as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                result = await session.call_tool("simulate_fomc_shock", {"scenario": "hawkish_50bps"})
                return tools, result

    def _assert_roundtrip(self, tools, result):
        self.assertEqual(EXPECTED_TOOLS, tools)
        self.assertFalse(NOT_MCP_TOOLS & tools)
        is_error = getattr(result, "is_error", getattr(result, "isError", False))  # v2 snake_case / v1 camelCase
        self.assertFalse(is_error)
        payload = json.loads(result.content[0].text)
        self.assertEqual(payload["status"], "success")
        self.assertLess(payload["total_pnl_dollar"], 0)  # a hawkish surprise hurts the preset

    async def _call(self, client_cm, name, args):
        async with client_cm as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                return await session.call_tool(name, args)

    def test_data_outage_is_an_mcp_error_and_never_fake_data(self):
        srv._cache.clear()
        self.addCleanup(srv._cache.clear)
        url = f"http://127.0.0.1:{self.port}/mcp"
        with mock.patch.object(srv, "_raw_download", side_effect=RuntimeError("yahoo down")):
            result = asyncio.run(asyncio.wait_for(
                self._call(http_client(url), "scan_quant_signals", {"ticker": "SPY"}), timeout=60))
        self.assertTrue(getattr(result, "is_error", getattr(result, "isError", False)))
        text = result.content[0].text
        self.assertIn("market data", text)
        self.assertNotIn("BUY", text)  # the old fallback invented a BUY signal at $560

    def test_base_url_shows_a_landing_page_for_browsers(self):
        status, data = self._http("GET", "/", headers={"Accept": "text/html"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["mcp_endpoint"], "/mcp")

    def test_mcp_also_works_at_the_base_url(self):
        # an add-on registered with the bare host must not get a 404
        url = f"http://127.0.0.1:{self.port}/"
        tools, result = asyncio.run(asyncio.wait_for(
            self._session_roundtrip(http_client(url)), timeout=60))
        self._assert_roundtrip(tools, result)

    def test_health(self):
        status, data = self._http("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["status"], "healthy")

    def test_streamable_http_handshake_and_tool_call(self):
        url = f"http://127.0.0.1:{self.port}/mcp"
        tools, result = asyncio.run(asyncio.wait_for(
            self._session_roundtrip(http_client(url)), timeout=60))
        self._assert_roundtrip(tools, result)

    def test_sse_handshake_and_tool_call(self):
        url = f"http://127.0.0.1:{self.port}/sse"
        tools, result = asyncio.run(asyncio.wait_for(
            self._session_roundtrip(sse_client(url)), timeout=60))
        self._assert_roundtrip(tools, result)

    def test_non_localhost_host_header_is_accepted(self):
        # The SDK auto-enables DNS-rebinding protection for host=127.0.0.1 and would answer
        # 421/403 to an ALB or custom-domain Host header, breaking real deployments.
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "host-header-test", "version": "0"}},
        })
        status, _ = self._http("POST", "/mcp", body=body, headers={
            "Host": "macropulse-alb-123.us-east-1.elb.amazonaws.com",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })
        self.assertNotIn(status, (403, 421), f"Host header rejected with HTTP {status}")

    def test_alexa_skill_webhook_shares_the_app(self):
        body = json.dumps({"version": "1.0", "session": {"new": True, "sessionId": "s", "user": {"userId": "u"}},
                           "request": {"type": "LaunchRequest", "requestId": "r", "locale": "en-US"}})
        status, data = self._http("POST", "/alexa/skill", body=body,
                                  headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertIn("outputSpeech", json.loads(data)["response"])


if __name__ == "__main__":
    unittest.main()
