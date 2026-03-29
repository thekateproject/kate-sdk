"""DiscoveryClient — manage autonomous discovery for Kate agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projectkate.models import DiscoveryAction, DiscoveryConfig

if TYPE_CHECKING:
    from projectkate.client import KateClient


class DiscoveryClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def get_config(self, agent_id: str) -> DiscoveryConfig:
        data = await self._client._request(
            "GET", f"/agents/{agent_id}/discovery/config"
        )
        return DiscoveryConfig(
            agent_id=str(data.get("agent_id", agent_id)),
            enabled=data.get("enabled", False),
            budget_tokens=data.get("budget_tokens", 0),
            auto_purchase=data.get("auto_purchase", False),
            max_price_tokens=data.get("max_price_tokens", 0),
            preferred_domains=data.get("preferred_domains", []),
        )

    async def configure(self, agent_id: str, **kwargs: object) -> DiscoveryConfig:
        data = await self._client._request(
            "PUT", f"/agents/{agent_id}/discovery/config", json=kwargs
        )
        return DiscoveryConfig(
            agent_id=str(data.get("agent_id", agent_id)),
            enabled=data.get("enabled", False),
            budget_tokens=data.get("budget_tokens", 0),
            auto_purchase=data.get("auto_purchase", False),
            max_price_tokens=data.get("max_price_tokens", 0),
            preferred_domains=data.get("preferred_domains", []),
        )

    async def run(self, agent_id: str) -> dict:
        """Trigger a discovery cycle. Returns 202 with status info."""
        return await self._client._request(
            "POST", f"/agents/{agent_id}/discovery/run"
        )

    async def list_actions(
        self, agent_id: str, limit: int = 20, offset: int = 0
    ) -> list[DiscoveryAction]:
        data = await self._client._request(
            "GET",
            f"/agents/{agent_id}/discovery/actions",
            params={"limit": limit, "offset": offset},
        )
        actions = data if isinstance(data, list) else data.get("actions", [])
        return [
            DiscoveryAction(
                id=str(a["id"]),
                agent_id=str(a.get("agent_id", agent_id)),
                action_type=a.get("action_type", ""),
                status=a.get("status", ""),
                summary=a.get("summary"),
                created_at=a.get("created_at"),
            )
            for a in actions
        ]
