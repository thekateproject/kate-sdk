"""ExtensionsClient — algorithmic extension access for Kate agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from projectkate._validation import validate_id

if TYPE_CHECKING:
    from projectkate.client import KateClient


@dataclass
class Extension:
    id: str
    target_function: str
    status: str
    version: int


class ExtensionsClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def get_active(self, agent_id: str, function_name: str) -> Extension | None:
        """GET /extensions/{agent_id}/{function_name} — returns None on 404."""
        validate_id(agent_id, "agent_id")
        try:
            data = await self._client._request(
                "GET", f"/extensions/{agent_id}/{function_name}"
            )
            return Extension(
                id=str(data["id"]),
                target_function=data["target_function"],
                status=data["status"],
                version=data["version"],
            )
        except Exception:
            return None

    async def execute(
        self, agent_id: str, function_name: str, input_data: dict
    ) -> dict:
        """POST /extensions/{agent_id}/execute"""
        validate_id(agent_id, "agent_id")
        data = await self._client._request(
            "POST",
            f"/extensions/{agent_id}/execute",
            json={"function_name": function_name, "input_data": input_data},
        )
        return data

    async def report_ab_result(
        self, agent_id: str, extension_id: str, enhanced_won: bool
    ) -> dict:
        """POST /extensions/{agent_id}/ab-result"""
        validate_id(agent_id, "agent_id")
        data = await self._client._request(
            "POST",
            f"/extensions/{agent_id}/ab-result",
            json={"extension_id": extension_id, "enhanced_won": enhanced_won},
        )
        return data
