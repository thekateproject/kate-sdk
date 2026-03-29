"""BriefsClient — knowledge brief access for Kate agents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from projectkate.models import Brief, BriefDiff, BriefVersion

if TYPE_CHECKING:
    from projectkate.client import KateClient


class BriefsClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def get(self, agent_id: str, version: str = "latest") -> Brief:
        params = {"version": version} if version != "latest" else {}
        data = await self._client._request(
            "GET", f"/agents/{agent_id}/brief", params=params
        )
        return Brief(
            agent_id=str(data.get("agent_id", agent_id)),
            version=data.get("version", ""),
            previous_version=data.get("previous_version"),
            compiled_at=data.get("compiled_at"),
            brief=data.get("brief", ""),
            gap_summary=data.get("gap_summary", []),
        )

    async def version(self, agent_id: str) -> BriefVersion:
        data = await self._client._request(
            "GET", f"/agents/{agent_id}/brief/version"
        )
        return BriefVersion(
            current_version=data["current_version"],
            latest_version=data["latest_version"],
            has_update=data.get("has_update", False),
            latest_compiled_at=data.get("latest_compiled_at"),
        )

    async def diff(
        self, agent_id: str, from_version: str, to_version: str
    ) -> BriefDiff:
        data = await self._client._request(
            "GET",
            f"/agents/{agent_id}/brief/diff",
            params={"from": from_version, "to": to_version},
        )
        return BriefDiff(
            from_version=data["from_version"],
            to_version=data["to_version"],
            summary=data.get("summary", ""),
            generated_at=data.get("generated_at"),
        )

    async def gaps(self, agent_id: str) -> list[str]:
        """Extract gap_summary from the latest brief."""
        brief = await self.get(agent_id)
        return brief.gap_summary

    async def export(
        self, agent_id: str, path: str | Path, version: str = "latest"
    ) -> Path:
        """Fetch the brief and write it to a local file."""
        brief = await self.get(agent_id, version=version)
        out = Path(path)
        out.write_text(brief.brief, encoding="utf-8")
        return out
