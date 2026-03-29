"""Tests for AgentsClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from projectkate.client import KateClient

_DUMMY_REQUEST = httpx.Request("GET", "http://kate.test:8000/test")

_AGENT_DATA = {
    "id": "agent-123",
    "name": "Test Agent",
    "domain": "finance",
    "objective": "Analyze markets",
    "status": "active",
    "owner_id": "user-1",
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z",
}


@pytest.fixture
def client():
    return KateClient(api_key="test-key", base_url="http://kate.test:8000")


@pytest.mark.asyncio
async def test_agents_list(client):
    mock_resp = httpx.Response(200, json=[_AGENT_DATA], request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        agents = await client.agents.list()

    assert len(agents) == 1
    assert agents[0].id == "agent-123"
    assert agents[0].name == "Test Agent"
    assert agents[0].domain == "finance"


@pytest.mark.asyncio
async def test_agents_get(client):
    mock_resp = httpx.Response(200, json=_AGENT_DATA, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        agent = await client.agents.get("agent-123")

    assert agent.id == "agent-123"
    assert agent.objective == "Analyze markets"


@pytest.mark.asyncio
async def test_agents_create(client):
    created = {**_AGENT_DATA, "id": "new-agent"}
    mock_resp = httpx.Response(201, json=created, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        agent = await client.agents.create("Test Agent", domain="finance")

    assert agent.id == "new-agent"
    call_args = mock_http.return_value.request.call_args
    assert call_args[1]["json"]["name"] == "Test Agent"
    assert call_args[1]["json"]["domain"] == "finance"


@pytest.mark.asyncio
async def test_agents_update(client):
    updated = {**_AGENT_DATA, "objective": "New objective"}
    mock_resp = httpx.Response(200, json=updated, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        agent = await client.agents.update("agent-123", objective="New objective")

    assert agent.objective == "New objective"


@pytest.mark.asyncio
async def test_agents_delete(client):
    mock_resp = httpx.Response(200, json={"detail": "deleted"}, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        result = await client.agents.delete("agent-123")

    assert result == {"detail": "deleted"}
