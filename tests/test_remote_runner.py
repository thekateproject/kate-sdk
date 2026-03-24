"""Tests for RemoteEvalRunner with mocked httpx."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from projectkate.context import SpanRecord
from projectkate.remote.runner import KateRemoteError, RemoteEvalRunner

_DUMMY_REQUEST = httpx.Request("POST", "http://kate.test:8000/test")


@pytest.fixture
def runner():
    return RemoteEvalRunner(
        api_url="http://kate.test:8000",
        api_key="test-key",
        agent_id="agent-123",
    )


@pytest.mark.asyncio
async def test_start_run(runner):
    mock_response = httpx.Response(
        200, json={"id": "run-456", "phoenix_project": "user_1_agent_123"},
        request=_DUMMY_REQUEST,
    )
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        result = await runner.start_run("run-456", "automatic")

    assert result == {"id": "run-456", "phoenix_project": "user_1_agent_123"}
    assert runner.phoenix_project == "user_1_agent_123"
    mock_client.return_value.post.assert_called_once_with(
        "/agents/agent-123/runs",
        json={"run_id": "run-456", "trigger": "automatic"},
    )


@pytest.mark.asyncio
async def test_start_run_returns_full_dict(runner):
    resp_data = {"id": "run-789", "phoenix_project": "proj_abc", "status": "pending"}
    mock_response = httpx.Response(200, json=resp_data, request=_DUMMY_REQUEST)
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        result = await runner.start_run("run-789", "manual")

    assert isinstance(result, dict)
    assert result["id"] == "run-789"
    assert result["phoenix_project"] == "proj_abc"
    assert runner.phoenix_project == "proj_abc"


@pytest.mark.asyncio
async def test_upload_spans(runner):
    mock_response = httpx.Response(200, json={"ok": True}, request=_DUMMY_REQUEST)
    spans = [
        SpanRecord(name="node1", input="in", output="out", span_kind="LLM", duration_ms=50.0),
    ]
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        await runner.upload_spans("run-456", spans)

    call_args = mock_client.return_value.post.call_args
    assert call_args[0][0] == "/agents/agent-123/runs/run-456/spans"
    payload = call_args[1]["json"]["spans"]
    assert len(payload) == 1
    assert payload[0]["name"] == "node1"
    assert payload[0]["input"] == "in"
    assert payload[0]["output"] == "out"


@pytest.mark.asyncio
async def test_complete_run(runner):
    mock_response = httpx.Response(
        200, json={"status": "completed"}, request=_DUMMY_REQUEST
    )
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        await runner.complete_run("run-456")

    mock_client.return_value.post.assert_called_once_with(
        "/agents/agent-123/runs/run-456/complete",
    )


@pytest.mark.asyncio
async def test_error_handling(runner):
    mock_response = httpx.Response(
        500, text="Internal Server Error", request=_DUMMY_REQUEST
    )
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        with pytest.raises(KateRemoteError, match="failed with status 500"):
            await runner.start_run("run-456", "automatic")


@pytest.mark.asyncio
async def test_agent_domain_in_ensure_agent():
    runner = RemoteEvalRunner(
        api_url="http://kate.test:8000",
        api_key="test-key",
        agent_id="",
        agent_name="my-agent",
        agent_domain="finance",
    )
    # Mock list agents returning empty → triggers create
    list_resp = httpx.Response(200, json=[], request=_DUMMY_REQUEST)
    create_resp = httpx.Response(200, json={"id": "new-agent-id"}, request=_DUMMY_REQUEST)

    with patch.object(runner, "_get_client") as mock_client:
        client = mock_client.return_value
        client.get = AsyncMock(return_value=list_resp)
        client.post = AsyncMock(return_value=create_resp)
        await runner.ensure_agent()

    # Verify domain is passed in the create call
    create_call = client.post.call_args
    assert create_call[1]["json"]["domain"] == "finance"
    assert runner.agent_id == "new-agent-id"


@pytest.mark.asyncio
async def test_poll_run_status_completes(runner):
    pending_resp = httpx.Response(
        200, json={"status": "pending"}, request=_DUMMY_REQUEST
    )
    completed_resp = httpx.Response(
        200, json={"status": "completed", "overall_score": 0.9}, request=_DUMMY_REQUEST
    )
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.get = AsyncMock(
            side_effect=[pending_resp, completed_resp]
        )
        result = await runner.poll_run_status("run-456", interval_seconds=0.01)

    assert result["status"] == "completed"
    assert result["overall_score"] == 0.9


@pytest.mark.asyncio
async def test_poll_run_status_timeout(runner):
    pending_resp = httpx.Response(
        200, json={"status": "pending"}, request=_DUMMY_REQUEST
    )
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.get = AsyncMock(return_value=pending_resp)
        with pytest.raises(KateRemoteError, match="did not complete"):
            await runner.poll_run_status(
                "run-456", interval_seconds=0.01, timeout_seconds=0.03,
            )
