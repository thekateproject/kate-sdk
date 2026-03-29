"""ArtifactsClient — artifact management for Kate sellers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from projectkate.models import Artifact, ArtifactAnalytics

if TYPE_CHECKING:
    from projectkate.client import KateClient


def _to_artifact(data: dict) -> Artifact:
    return Artifact(
        id=str(data["id"]),
        title=data.get("title", ""),
        domain=data.get("domain", ""),
        status=data.get("status", ""),
        user_id=str(data["user_id"]) if data.get("user_id") else None,
        agent_id=str(data["agent_id"]) if data.get("agent_id") else None,
        description=data.get("description", ""),
        hosting_type=data.get("hosting_type"),
        creation_type=data.get("creation_type"),
        visibility=data.get("visibility"),
        price_tokens=data.get("price_tokens", 0),
        per_query_tokens=data.get("per_query_tokens", 0),
        version=data.get("version", "1.0.0"),
        quality_score=data.get("quality_score"),
        covers_generated=data.get("covers_generated", False),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


class ArtifactsClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def list(self) -> list[Artifact]:
        data = await self._client._request("GET", "/artifacts")
        items = data if isinstance(data, list) else data.get("artifacts", [])
        return [_to_artifact(a) for a in items]

    async def get(self, artifact_id: str) -> Artifact:
        data = await self._client._request("GET", f"/artifacts/{artifact_id}")
        return _to_artifact(data)

    async def create_from_agent(self, agent_id: str) -> Artifact:
        data = await self._client._request(
            "POST", "/artifacts/extract-from-agent", json={"agent_id": agent_id}
        )
        return _to_artifact(data)

    async def upload(
        self,
        name: str,
        file_path: str | Path,
        agent_id: str | None = None,
        domain: str = "general",
    ) -> Artifact:
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            form_data = {"title": name, "domain": domain}
            if agent_id:
                form_data["agent_id"] = agent_id
            data = await self._client._request(
                "POST", "/artifacts/upload", data=form_data, files=files
            )
        return _to_artifact(data)

    async def analyze_coverage(self, artifact_id: str) -> Artifact:
        data = await self._client._request(
            "POST", f"/artifacts/{artifact_id}/analyze-coverage"
        )
        return _to_artifact(data)

    async def publish(self, artifact_id: str) -> Artifact:
        """Publish an artifact. Auto-generates cover if missing."""
        data = await self._client._request(
            "POST", f"/artifacts/{artifact_id}/publish"
        )
        return _to_artifact(data)

    async def analytics(self, artifact_id: str) -> ArtifactAnalytics:
        data = await self._client._request(
            "GET", f"/artifacts/{artifact_id}/analytics"
        )
        return ArtifactAnalytics(
            subscriber_count=data.get("subscriber_count", 0),
            total_queries=data.get("total_queries", 0),
            tokens_earned=data.get("tokens_earned", 0),
        )
