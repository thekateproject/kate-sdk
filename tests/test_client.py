"""Tests for KateClient core + agents sub-client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from projectkate.client import KateClient
from projectkate.remote.runner import KateRemoteError

_DUMMY_REQUEST = httpx.Request("GET", "http://kate.test:8000/test")


@pytest.fixture
def client():
    return KateClient(api_key="test-key", base_url="http://kate.test:8000")


@pytest.mark.asyncio
async def test_request_get(client):
    mock_resp = httpx.Response(200, json={"ok": True}, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        result = await client._request("GET", "/test")

    assert result == {"ok": True}
    mock_http.return_value.request.assert_called_once_with(
        "GET", "/test", json=None, params=None, data=None, files=None,
    )


@pytest.mark.asyncio
async def test_request_auth_error(client):
    mock_resp = httpx.Response(
        401, json={"detail": "Invalid key"}, request=_DUMMY_REQUEST
    )
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        with pytest.raises(KateRemoteError, match="Authentication failed"):
            await client._request("GET", "/agents")


@pytest.mark.asyncio
async def test_request_not_found(client):
    mock_resp = httpx.Response(
        404, json={"detail": "Not found"}, request=_DUMMY_REQUEST
    )
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        with pytest.raises(KateRemoteError, match="not found"):
            await client._request("GET", "/agents/missing")


@pytest.mark.asyncio
async def test_context_manager():
    client = KateClient(api_key="test-key", base_url="http://kate.test:8000")
    async with client as c:
        assert c is client


@pytest.mark.asyncio
async def test_sub_clients_lazy(client):
    """Sub-clients are created on first access."""
    agents1 = client.agents
    agents2 = client.agents
    assert agents1 is agents2  # Same instance

    # Different sub-client types
    assert client.briefs is not client.agents
    assert client.wallet is not client.agents
