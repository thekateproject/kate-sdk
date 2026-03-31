"""DiscoveryClient — manage autonomous discovery for Kate agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projectkate._validation import validate_id
from projectkate.models import DiscoveryAction, DiscoveryConfig

if TYPE_CHECKING:
    from projectkate.client import KateClient


class DiscoveryClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def get_config(self, agent_id: str) -> DiscoveryConfig:
        validate_id(agent_id, "agent_id")
        data = await self._client._request(
            "GET", f"/agents/{agent_id}/discovery/config"
        )
        return DiscoveryConfig(
            agent_id=str(data.get("agent_id", agent_id)),
            mode=data.get("mode", "manual"),
            interval_hours=data.get("interval_hours", 24),
            max_tokens_per_action=data.get("max_tokens_per_action", 500),
            daily_action_limit=data.get("daily_action_limit", 3),
            min_compatibility_score=data.get("min_compatibility_score", 0.5),
            max_candidates_per_gap=data.get("max_candidates_per_gap", 3),
            last_run_at=data.get("last_run_at"),
            actions_today=data.get("actions_today", 0),
        )

    async def configure(self, agent_id: str, **kwargs: object) -> DiscoveryConfig:
        validate_id(agent_id, "agent_id")
        data = await self._client._request(
            "PUT", f"/agents/{agent_id}/discovery/config", json=kwargs
        )
        return DiscoveryConfig(
            agent_id=str(data.get("agent_id", agent_id)),
            mode=data.get("mode", "manual"),
            interval_hours=data.get("interval_hours", 24),
            max_tokens_per_action=data.get("max_tokens_per_action", 500),
            daily_action_limit=data.get("daily_action_limit", 3),
            min_compatibility_score=data.get("min_compatibility_score", 0.5),
            max_candidates_per_gap=data.get("max_candidates_per_gap", 3),
            last_run_at=data.get("last_run_at"),
            actions_today=data.get("actions_today", 0),
        )

    async def run(self, agent_id: str) -> dict:
        """Trigger a discovery cycle. Returns 202 with status info."""
        validate_id(agent_id, "agent_id")
        return await self._client._request(
            "POST", f"/agents/{agent_id}/discovery/run"
        )

    async def list_actions(
        self, agent_id: str, limit: int = 20, offset: int = 0
    ) -> list[DiscoveryAction]:
        validate_id(agent_id, "agent_id")
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
