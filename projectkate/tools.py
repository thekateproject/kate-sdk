"""ToolsClient — discover and execute KATE marketplace tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projectkate._validation import validate_id
from projectkate.models import ToolCredentialStatus, ToolResult
from projectkate.remote.runner import (
    KateBalanceError,
    KateCredentialError,
    KateRemoteError,
    KateToolError,
)

if TYPE_CHECKING:
    from projectkate.client import KateClient

__all__ = ["ToolsClient", "KateBalanceError", "KateCredentialError", "KateToolError"]


class ToolsClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def list(self, agent_id: str, format: str = "openai") -> list[dict]:
        """GET /tools/{agent_id} — list available tools."""
        validate_id(agent_id, "agent_id")
        data = await self._client._request("GET", f"/tools/{agent_id}")
        tools = data if isinstance(data, list) else []
        if format == "openai":
            return [{k: v for k, v in t.items() if k != "metadata"} for t in tools]
        return tools

    async def execute(
        self, agent_id: str, tool_name: str, input_data: dict | None = None,
    ) -> ToolResult:
        """POST /tools/{agent_id}/execute — execute a tool and return ToolResult."""
        validate_id(agent_id, "agent_id")
        try:
            data = await self._client._request(
                "POST",
                f"/tools/{agent_id}/execute",
                json={"tool_name": tool_name, "input": input_data or {}},
            )
        except KateRemoteError as exc:
            msg = str(exc)
            if "status 428" in msg:
                raise KateCredentialError(msg) from exc
            if "status 402" in msg:
                raise KateBalanceError(msg) from exc
            if "status 502" in msg or "status 504" in msg:
                raise KateToolError(msg) from exc
            raise
        return ToolResult(
            success=data.get("success", True),
            output=data.get("output"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0),
            tokens_charged=data.get("tokens_charged", 0),
        )

    async def status(self, agent_id: str) -> list[ToolCredentialStatus]:
        """GET /tools/{agent_id}/status — credential status for all tools."""
        validate_id(agent_id, "agent_id")
        data = await self._client._request("GET", f"/tools/{agent_id}/status")
        items = data if isinstance(data, list) else []
        return [
            ToolCredentialStatus(
                tool_name=item.get("tool_name", ""),
                artifact_id=str(item.get("artifact_id", "")),
                status=item.get("status", "active"),
                missing_keys=item.get("missing_keys", []),
            )
            for item in items
        ]
