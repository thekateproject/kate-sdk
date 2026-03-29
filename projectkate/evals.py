"""EvalsClient — read-only access to eval intelligence summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projectkate.models import EvalSummary

if TYPE_CHECKING:
    from projectkate.client import KateClient


class EvalsClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def summary(self, agent_id: str) -> EvalSummary:
        data = await self._client._request("GET", f"/agents/{agent_id}/eval/summary")
        return EvalSummary(
            intelligence_summary=data["intelligence_summary"],
            updated_at=data["updated_at"],
            overall_score=data.get("overall_score"),
            recommendations=data.get("recommendations", []),
            regression_detected=data.get("regression_detected", False),
        )
