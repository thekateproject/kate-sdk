"""Tests for ExtensionsClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from projectkate.client import KateClient
from projectkate.extensions import Extension

_DUMMY_REQUEST = httpx.Request("GET", "https://kate.test:8000/test")

_EXTENSION_DATA = {
    "id": "ext-111-222-333-444",
    "target_function": "summarize",
    "status": "active",
    "version": 2,
}


@pytest.fixture
def client():
    return KateClient(api_key="test-key", base_url="https://kate.test:8000")


@pytest.mark.asyncio
async def test_get_active_found(client):
    mock_resp = httpx.Response(200, json=_EXTENSION_DATA, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        ext = await client.extensions.get_active(
            "00000000-0000-0000-0000-000000000001", "summarize"
        )

    assert ext is not None
    assert isinstance(ext, Extension)
    assert ext.id == "ext-111-222-333-444"
    assert ext.target_function == "summarize"
    assert ext.status == "active"
    assert ext.version == 2


@pytest.mark.asyncio
async def test_get_active_not_found(client):
    mock_resp = httpx.Response(
        404,
        json={"detail": "Not found"},
        request=_DUMMY_REQUEST,
    )
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        ext = await client.extensions.get_active(
            "00000000-0000-0000-0000-000000000001", "summarize"
        )

    assert ext is None


@pytest.mark.asyncio
async def test_execute_success(client):
    execute_data = {"success": True, "output": "enhanced result", "error": None, "logs": []}
    mock_resp = httpx.Response(200, json=execute_data, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        result = await client.extensions.execute(
            "00000000-0000-0000-0000-000000000001",
            "summarize",
            {"original_input": "text", "original_output": "summary"},
        )

    assert result["success"] is True
    assert result["output"] == "enhanced result"


@pytest.mark.asyncio
async def test_report_ab_result(client):
    ab_data = {
        "id": "ext-111-222-333-444",
        "ab_trials_total": 10,
        "ab_win_rate": 0.7,
        "ab_auto_decision": None,
        "status": "active",
    }
    mock_resp = httpx.Response(200, json=ab_data, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        result = await client.extensions.report_ab_result(
            "00000000-0000-0000-0000-000000000001",
            "ext-111-222-333-444",
            enhanced_won=True,
        )

    assert result["ab_trials_total"] == 10
    assert result["ab_win_rate"] == 0.7
