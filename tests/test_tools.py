"""Tests for tool discovery, execution, and tracing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from projectkate._state import KateSDK
from projectkate.client import KateClient
from projectkate.context import SpanRecord
from projectkate.models import ToolCredentialStatus, ToolResult
from projectkate.remote.runner import (
    KateBalanceError,
    KateCredentialError,
    KateRemoteError,
    KateToolError,
    RemoteEvalRunner,
)
from projectkate.tools import ToolsClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {"name": "seo_brief", "description": "Generate SEO brief", "parameters": {}},
        "metadata": {"artifact_id": "abc123", "price": 10},
    },
    {
        "type": "function",
        "function": {"name": "keyword_research", "description": "Find keywords", "parameters": {}},
        "metadata": {"artifact_id": "def456", "price": 5},
    },
]


def _mock_response(status_code: int = 200, json_data=None, text: str = "") -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text or json.dumps(json_data or {})
    return resp


# ---------------------------------------------------------------------------
# ToolsClient tests
# ---------------------------------------------------------------------------


class TestToolsClient:
    @pytest.fixture()
    def client(self):
        c = KateClient.__new__(KateClient)
        c._api_key = "test"
        c._base_url = "https://localhost"
        c._timeout = 30.0
        c._http = None
        c._agents = c._evals = c._discovery = c._briefs = None
        c._artifacts = c._wallet = c._runs = c._tools = None
        return c

    @pytest.fixture()
    def tools_client(self, client):
        return ToolsClient(client)

    @pytest.mark.asyncio
    async def test_tools_list_openai_format(self, tools_client, client):
        client._request = AsyncMock(return_value=SAMPLE_TOOLS)
        result = await tools_client.list("00000000-0000-0000-0000-000000000001", format="openai")
        assert len(result) == 2
        for t in result:
            assert "metadata" not in t
        assert result[0]["function"]["name"] == "seo_brief"

    @pytest.mark.asyncio
    async def test_tools_list_raw_format(self, tools_client, client):
        client._request = AsyncMock(return_value=SAMPLE_TOOLS)
        result = await tools_client.list("00000000-0000-0000-0000-000000000001", format="raw")
        assert len(result) == 2
        assert "metadata" in result[0]
        assert result[0]["metadata"]["artifact_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_tools_execute_success(self, tools_client, client):
        client._request = AsyncMock(return_value={
            "success": True,
            "output": {"brief": "SEO content"},
            "execution_time_ms": 150,
            "tokens_charged": 10,
        })
        result = await tools_client.execute("00000000-0000-0000-0000-000000000001", "seo_brief", {"topic": "AI"})
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == {"brief": "SEO content"}
        assert result.tokens_charged == 10

    @pytest.mark.asyncio
    async def test_tools_execute_credential_error(self, tools_client, client):
        client._request = AsyncMock(
            side_effect=KateRemoteError("[KATE] execute failed with status 428. Detail: missing creds")
        )
        with pytest.raises(KateCredentialError):
            await tools_client.execute("00000000-0000-0000-0000-000000000001", "seo_brief")

    @pytest.mark.asyncio
    async def test_tools_execute_balance_error(self, tools_client, client):
        client._request = AsyncMock(
            side_effect=KateRemoteError("[KATE] execute failed with status 402. Detail: insufficient balance")
        )
        with pytest.raises(KateBalanceError):
            await tools_client.execute("00000000-0000-0000-0000-000000000001", "seo_brief")

    @pytest.mark.asyncio
    async def test_tools_execute_tool_error(self, tools_client, client):
        client._request = AsyncMock(
            side_effect=KateRemoteError("[KATE] execute failed with status 502. Internal server error")
        )
        with pytest.raises(KateToolError):
            await tools_client.execute("00000000-0000-0000-0000-000000000001", "seo_brief")

    @pytest.mark.asyncio
    async def test_tools_status(self, tools_client, client):
        client._request = AsyncMock(return_value=[
            {"tool_name": "seo_brief", "artifact_id": "abc123", "status": "active", "missing_keys": []},
            {"tool_name": "keyword_research", "artifact_id": "def456", "status": "missing", "missing_keys": ["api_key"]},
        ])
        result = await tools_client.status("00000000-0000-0000-0000-000000000001")
        assert len(result) == 2
        assert isinstance(result[0], ToolCredentialStatus)
        assert result[0].tool_name == "seo_brief"
        assert result[1].missing_keys == ["api_key"]


# ---------------------------------------------------------------------------
# RemoteEvalRunner tests
# ---------------------------------------------------------------------------


class TestRunnerTools:
    @pytest.fixture()
    def runner(self):
        return RemoteEvalRunner(
            api_url="https://localhost",
            api_key="test-key",
            agent_id="00000000-0000-0000-0000-000000000001",
        )

    @pytest.mark.asyncio
    async def test_runner_get_tools(self, runner):
        mock_resp = _mock_response(200, SAMPLE_TOOLS)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        runner._get_client = MagicMock(return_value=mock_client)

        result = await runner.get_tools(format="openai")
        assert len(result) == 2
        for t in result:
            assert "metadata" not in t

    @pytest.mark.asyncio
    async def test_runner_call_tool_success(self, runner):
        mock_resp = _mock_response(200, {"success": True, "output": "done", "tokens_charged": 5})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        runner._get_client = MagicMock(return_value=mock_client)

        result = await runner.call_tool("seo_brief", {"topic": "AI"})
        assert result["success"] is True
        assert result["tokens_charged"] == 5

    @pytest.mark.asyncio
    async def test_runner_call_tool_428(self, runner):
        mock_resp = _mock_response(428, {"detail": "missing credentials"})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        runner._get_client = MagicMock(return_value=mock_client)

        with pytest.raises(KateCredentialError):
            await runner.call_tool("seo_brief")

    @pytest.mark.asyncio
    async def test_runner_call_tool_402(self, runner):
        mock_resp = _mock_response(402, {"detail": "insufficient balance"})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        runner._get_client = MagicMock(return_value=mock_client)

        with pytest.raises(KateBalanceError):
            await runner.call_tool("seo_brief")

    @pytest.mark.asyncio
    async def test_runner_get_tools_unreachable(self, runner):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        runner._get_client = MagicMock(return_value=mock_client)

        result = await runner.get_tools()
        assert result == []


# ---------------------------------------------------------------------------
# Module-level function tests
# ---------------------------------------------------------------------------


class TestModuleLevelTools:
    def setup_method(self):
        KateSDK.reset()

    def teardown_method(self):
        KateSDK.reset()

    @pytest.mark.asyncio
    async def test_call_tool_traces_span(self):
        sdk = KateSDK.get()
        mock_runner = AsyncMock()
        mock_runner.call_tool = AsyncMock(return_value={
            "success": True, "output": "result", "execution_time_ms": 100, "tokens_charged": 5,
        })
        sdk._remote_runner = mock_runner
        sdk.mode = "remote"

        # Set up a run context to capture spans
        from projectkate.context import RunContext
        sdk._active_ctx = RunContext()

        import projectkate
        result = await projectkate.call_tool("seo_brief", {"topic": "AI"})

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert len(sdk._active_ctx.spans) == 1
        span = sdk._active_ctx.spans[0]
        assert span.span_kind == "TOOL"
        assert span.name == "seo_brief"
        assert span.error is None

    @pytest.mark.asyncio
    async def test_make_kate_tool_executor(self):
        sdk = KateSDK.get()
        mock_runner = AsyncMock()
        mock_runner.call_tool = AsyncMock(return_value={
            "success": True, "output": "done", "execution_time_ms": 50, "tokens_charged": 3,
        })
        sdk._remote_runner = mock_runner
        sdk.mode = "remote"

        import projectkate
        executors = projectkate.make_kate_tool_executor(SAMPLE_TOOLS)

        assert "seo_brief" in executors
        assert "keyword_research" in executors
        assert callable(executors["seo_brief"])

        result = await executors["seo_brief"]({"topic": "test"})
        assert isinstance(result, ToolResult)
        mock_runner.call_tool.assert_called_once_with("seo_brief", {"topic": "test"})

    @pytest.mark.asyncio
    async def test_get_tools_no_init_raises(self):
        import projectkate
        with pytest.raises(RuntimeError, match="requires remote mode"):
            await projectkate.get_tools()
