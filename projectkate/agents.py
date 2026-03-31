"""AgentsClient — CRUD operations for Kate agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projectkate._validation import validate_id
from projectkate.models import Agent

if TYPE_CHECKING:
    from projectkate.client import KateClient


def _to_agent(data: dict) -> Agent:
    return Agent(
        id=str(data["id"]),
        name=data.get("name", ""),
        domain=data.get("domain", ""),
        objective=data.get("objective"),
        status=data.get("status"),
        owner_id=str(data["owner_id"]) if data.get("owner_id") else None,
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


class AgentsClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def list(self) -> list[Agent]:
        data = await self._client._request("GET", "/agents")
        return [_to_agent(a) for a in data]

    async def get(self, agent_id: str) -> Agent:
        validate_id(agent_id, "agent_id")
        data = await self._client._request("GET", f"/agents/{agent_id}")
        return _to_agent(data)

    async def create(
        self,
        name: str,
        domain: str = "general",
        objective: str | None = None,
    ) -> Agent:
        body: dict = {"name": name, "domain": domain}
        if objective:
            body["objective"] = objective
        data = await self._client._request("POST", "/agents", json=body)
        return _to_agent(data)

    async def update(self, agent_id: str, **kwargs: object) -> Agent:
        validate_id(agent_id, "agent_id")
        data = await self._client._request("PATCH", f"/agents/{agent_id}", json=kwargs)
        return _to_agent(data)

    async def delete(self, agent_id: str) -> dict:
        validate_id(agent_id, "agent_id")
        return await self._client._request("DELETE", f"/agents/{agent_id}")
