"""Tests for ArtifactsClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from projectkate.client import KateClient

_DUMMY_REQUEST = httpx.Request("GET", "https://kate.test:8000/test")

_ARTIFACT_DATA = {
    "id": "art-123",
    "title": "SEO Guide",
    "domain": "seo",
    "status": "draft",
    "user_id": "user-1",
    "agent_id": "agent-1",
    "description": "A guide to SEO",
    "hosting_type": "kate_hosted",
    "visibility": "listed",
    "price_tokens": 100,
    "per_query_tokens": 5,
    "version": "1.0.0",
    "quality_score": 0.85,
    "covers_generated": True,
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z",
}


@pytest.fixture
def client():
    return KateClient(api_key="test-key", base_url="https://kate.test:8000")


@pytest.mark.asyncio
async def test_artifacts_list(client):
    mock_resp = httpx.Response(200, json=[_ARTIFACT_DATA], request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        artifacts = await client.artifacts.list()

    assert len(artifacts) == 1
    assert artifacts[0].title == "SEO Guide"
    assert artifacts[0].quality_score == 0.85


@pytest.mark.asyncio
async def test_artifacts_get(client):
    mock_resp = httpx.Response(200, json=_ARTIFACT_DATA, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        artifact = await client.artifacts.get("art-123")

    assert artifact.id == "art-123"
    assert artifact.price_tokens == 100


@pytest.mark.asyncio
async def test_artifacts_publish(client):
    published = {**_ARTIFACT_DATA, "status": "listed"}
    mock_resp = httpx.Response(200, json=published, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        artifact = await client.artifacts.publish("art-123")

    assert artifact.status == "listed"


@pytest.mark.asyncio
async def test_artifacts_analytics(client):
    analytics_data = {
        "subscriber_count": 12,
        "total_queries": 340,
        "tokens_earned": 1700,
    }
    mock_resp = httpx.Response(200, json=analytics_data, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        analytics = await client.artifacts.analytics("art-123")

    assert analytics.subscriber_count == 12
    assert analytics.total_queries == 340
    assert analytics.tokens_earned == 1700


@pytest.mark.asyncio
async def test_artifacts_create_from_agent(client):
    mock_resp = httpx.Response(201, json=_ARTIFACT_DATA, request=_DUMMY_REQUEST)
    with patch.object(client, "_get_http") as mock_http:
        mock_http.return_value.request = AsyncMock(return_value=mock_resp)
        artifact = await client.artifacts.create_from_agent("agent-1")

    assert artifact.id == "art-123"
    call_args = mock_http.return_value.request.call_args
    assert call_args[1]["json"]["agent_id"] == "agent-1"
