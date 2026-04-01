"""Tests for BriefsClient."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from projectkate.client import KateClient

_DUMMY_REQUEST = httpx.Request("GET", "https://kate.test:8000/test")

_BRIEF_DATA = {
    "agent_id": "agent-123",
    "version": "kb_3",
    "previous_version": "kb_2",
    "compiled_at": "2025-06-01T00:00:00Z",
    "brief": "This agent specializes in SEO content analysis.",
    "gap_summary": ["Missing coverage on technical SEO", "No backlink analysis"],
}


@pytest.fixture
def client():
    return KateClient(api_key="test-key", base_url="https://kate.test:8000")


@pytest.mark.asyncio
async def test_briefs_get(client):
    mock_resp = httpx.Response(200, json=_BRIEF_DATA, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        brief = await client.briefs.get("agent-123")

    assert brief.version == "kb_3"
    assert brief.brief == "This agent specializes in SEO content analysis."
    assert len(brief.gap_summary) == 2


@pytest.mark.asyncio
async def test_briefs_get_specific_version(client):
    mock_resp = httpx.Response(200, json=_BRIEF_DATA, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        await client.briefs.get("agent-123", version="kb_2")

    call_args = mock_http.return_value.request.call_args
    assert call_args[1]["params"] == {"version": "kb_2"}


@pytest.mark.asyncio
async def test_briefs_version(client):
    version_data = {
        "current_version": "kb_3",
        "latest_version": "kb_3",
        "has_update": False,
        "latest_compiled_at": "2025-06-01T00:00:00Z",
    }
    mock_resp = httpx.Response(200, json=version_data, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        v = await client.briefs.version("agent-123")

    assert v.current_version == "kb_3"
    assert v.has_update is False


@pytest.mark.asyncio
async def test_briefs_diff(client):
    diff_data = {
        "from_version": "kb_2",
        "to_version": "kb_3",
        "summary": "Added SEO analysis coverage.",
        "generated_at": "2025-06-01T00:00:00Z",
    }
    mock_resp = httpx.Response(200, json=diff_data, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        diff = await client.briefs.diff("agent-123", "kb_2", "kb_3")

    assert diff.summary == "Added SEO analysis coverage."
    call_args = mock_http.return_value.request.call_args
    assert call_args[1]["params"] == {"from": "kb_2", "to": "kb_3"}


@pytest.mark.asyncio
async def test_briefs_gaps(client):
    mock_resp = httpx.Response(200, json=_BRIEF_DATA, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        gaps = await client.briefs.gaps("agent-123")

    assert gaps == ["Missing coverage on technical SEO", "No backlink analysis"]


@pytest.mark.asyncio
async def test_briefs_export(client):
    mock_resp = httpx.Response(200, json=_BRIEF_DATA, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = await client.briefs.export("agent-123", f.name)

    assert path.read_text() == "This agent specializes in SEO content analysis."
