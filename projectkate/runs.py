"""RunsClient — read-only access to agent run history."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projectkate._validation import validate_id
from projectkate.models import RunSummary

if TYPE_CHECKING:
    from projectkate.client import KateClient


class RunsClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def list(
        self, agent_id: str, limit: int = 5, offset: int = 0
    ) -> list[RunSummary]:
        validate_id(agent_id, "agent_id")
        data = await self._client._request(
            "GET",
            f"/agents/{agent_id}/runs",
            params={"limit": limit, "offset": offset},
        )
        items = data if isinstance(data, list) else data.get("runs", [])
        return [
            RunSummary(
                id=str(r["id"]),
                agent_id=str(r.get("agent_id", agent_id)),
                status=r.get("status", ""),
                trigger=r.get("trigger"),
                overall_score=r.get("overall_score"),
                created_at=r.get("created_at"),
                completed_at=r.get("completed_at"),
            )
            for r in items
        ]
