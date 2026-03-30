"""Tests for DiscoveryClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from projectkate.client import KateClient

_DUMMY_REQUEST = httpx.Request("GET", "http://kate.test:8000/test")


@pytest.fixture
def client():
    return KateClient(api_key="test-key", base_url="http://kate.test:8000")


@pytest.mark.asyncio
async def test_discovery_get_config(client):
    config_data = {
        "agent_id": "agent-123",
        "mode": "autonomous",
        "max_tokens_per_action": 500,
        "interval_hours": 24,
        "daily_action_limit": 3,
        "min_compatibility_score": 0.5,
        "max_candidates_per_gap": 3,
    }
    mock_resp = httpx.Response(200, json=config_data, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        config = await client.discovery.get_config("agent-123")

    assert config.mode == "autonomous"
    assert config.max_tokens_per_action == 500
    assert config.max_candidates_per_gap == 3


@pytest.mark.asyncio
async def test_discovery_configure(client):
    updated = {
        "agent_id": "agent-123",
        "mode": "autonomous",
        "max_tokens_per_action": 1000,
        "interval_hours": 12,
        "daily_action_limit": 5,
        "min_compatibility_score": 0.7,
        "max_candidates_per_gap": 5,
    }
    mock_resp = httpx.Response(200, json=updated, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        config = await client.discovery.configure("agent-123", max_tokens_per_action=1000)

    assert config.max_tokens_per_action == 1000
    call_args = mock_http.return_value.request.call_args
    assert call_args[1]["json"]["max_tokens_per_action"] == 1000


@pytest.mark.asyncio
async def test_discovery_run(client):
    mock_resp = httpx.Response(
        202, json={"status": "accepted", "message": "Discovery started"},
        request=_DUMMY_REQUEST,
    )
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        result = await client.discovery.run("agent-123")

    assert result["status"] == "accepted"


@pytest.mark.asyncio
async def test_discovery_list_actions(client):
    actions_data = [
        {
            "id": "action-1",
            "agent_id": "agent-123",
            "action_type": "search",
            "status": "completed",
            "summary": {"candidates_found": 5},
            "created_at": "2025-06-01T00:00:00Z",
        },
    ]
    mock_resp = httpx.Response(200, json=actions_data, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        actions = await client.discovery.list_actions("agent-123")

    assert len(actions) == 1
    assert actions[0].action_type == "search"
    assert actions[0].summary == {"candidates_found": 5}
