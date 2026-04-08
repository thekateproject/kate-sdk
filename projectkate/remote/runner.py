"""RemoteEvalRunner — uploads spans to KATE API server."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import httpx

from projectkate._validation import validate_id

if TYPE_CHECKING:
    from projectkate.context import SpanRecord

logger = logging.getLogger(__name__)


class KateRemoteError(Exception):
    """Human-readable error for KATE API failures."""


class KateToolError(KateRemoteError):
    """Tool execution failed on the server (502/504)."""


class KateCredentialError(KateRemoteError):
    """Tool requires credentials not yet configured (428)."""


class KateBalanceError(KateRemoteError):
    """Insufficient token balance to execute tool (402)."""


class RemoteEvalRunner:
    def __init__(
        self, api_url: str, api_key: str, agent_id: str, agent_name: str | None = None,
        agent_objective: str | None = None, agent_domain: str | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self._api_key = api_key
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_objective = agent_objective
        self.agent_domain = agent_domain or "general"
        self._client: httpx.AsyncClient | None = None
        if (
            not api_url.startswith("https://")
            and "localhost" not in api_url
            and "127.0.0.1" not in api_url
        ):
            raise ValueError(
                "KATE API URL must use HTTPS. "
                "Plaintext HTTP is only allowed for localhost/127.0.0.1."
            )

    def __repr__(self) -> str:
        return f"RemoteEvalRunner(api_url={self.api_url!r}, agent_id={self.agent_id!r})"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                headers={"x-api-key": self._api_key},
                timeout=30.0,
            )
        return self._client

    @staticmethod
    def _handle_response(resp: httpx.Response, operation: str) -> None:
        """Raise a clear error for common HTTP failures."""
        if resp.is_success:
            return
        code = resp.status_code
        detail = ""
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text

        if code == 401:
            raise KateRemoteError(
                f"[KATE] Authentication failed during {operation}. "
                "Check that your KATE_API_KEY is correct and not expired."
            )
        if code == 404:
            raise KateRemoteError(
                f"[KATE] Resource not found during {operation} (404). "
                f"Detail: {detail}"
            )
        if code == 422:
            raise KateRemoteError(
                f"[KATE] Validation error during {operation} (422). "
                f"Detail: {detail}"
            )
        if code >= 500:
            raise KateRemoteError(
                f"[KATE] {operation} failed with status {code}. Internal server error"
            )
        raise KateRemoteError(
            f"[KATE] {operation} failed with status {code}. Detail: {detail}"
        )

    async def ensure_agent(self) -> None:
        """Auto-register agent by name if agent_id is empty."""
        if self.agent_id:
            return
        if not self.agent_name:
            raise KateRemoteError(
                "[KATE] No agent_id or agent_name provided. "
                "Pass agent_id= or agent_name= to projectkate.init()."
            )
        # Check if agent with this name already exists
        resp = await self._get_client().get("/agents")
        self._handle_response(resp, "list agents")
        for agent in resp.json():
            if agent.get("name") == self.agent_name:
                self.agent_id = agent["id"]
                # Update objective if provided and different
                if self.agent_objective and agent.get("objective") != self.agent_objective:
                    resp = await self._get_client().patch(
                        f"/agents/{self.agent_id}",
                        json={"objective": self.agent_objective},
                    )
                    self._handle_response(resp, "update agent objective")
                return
        # Create new agent
        body: dict = {"name": self.agent_name, "domain": self.agent_domain}
        if self.agent_objective:
            body["objective"] = self.agent_objective
        resp = await self._get_client().post("/agents", json=body)
        self._handle_response(resp, "auto-register agent")
        self.agent_id = resp.json()["id"]

    async def start_run(self, run_id: str, trigger: str = "manual") -> dict:
        """POST /agents/{id}/runs — create a new eval run. Returns full response dict."""
        validate_id(run_id, "run_id")
        await self.ensure_agent()
        resp = await self._get_client().post(
            f"/agents/{self.agent_id}/runs",
            json={"run_id": run_id, "trigger": trigger},
        )
        self._handle_response(resp, "start run")
        return resp.json()

    async def upload_spans(self, run_id: str, spans: list[SpanRecord]) -> None:
        """POST /agents/{id}/runs/{run_id}/spans — upload traced spans."""
        validate_id(run_id, "run_id")
        payload = [
            {
                "name": s.name,
                "span_kind": s.span_kind,
                "input": s.input,
                "output": s.output,
                "latency_ms": round(s.duration_ms),
                "error": s.error,
                "model": s.model,
                "token_count": s.token_count,
                "docstring": s.docstring,
            }
            for s in spans
        ]
        resp = await self._get_client().post(
            f"/agents/{self.agent_id}/runs/{run_id}/spans",
            json={"spans": payload},
        )
        self._handle_response(resp, "upload spans")

    async def complete_run(self, run_id: str) -> None:
        """POST /agents/{id}/runs/{run_id}/complete — trigger server-side eval."""
        validate_id(run_id, "run_id")
        resp = await self._get_client().post(
            f"/agents/{self.agent_id}/runs/{run_id}/complete",
        )
        self._handle_response(resp, "complete run")

    async def poll_run_status(
        self, run_id: str, *, interval_seconds: float = 2.0, timeout_seconds: float = 300.0,
    ) -> dict:
        """Poll GET /agents/{id}/runs/{run_id} until terminal status."""
        validate_id(run_id, "run_id")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            resp = await self._get_client().get(
                f"/agents/{self.agent_id}/runs/{run_id}",
            )
            self._handle_response(resp, "get run status")
            data = resp.json()
            if data.get("status") in ("completed", "failed"):
                return data
            await asyncio.sleep(interval_seconds)
        raise KateRemoteError(
            f"[KATE] Run {run_id} did not complete within {timeout_seconds}s"
        )

    async def get_tools(self, format: str = "openai") -> list[dict]:
        """GET /tools/{agent_id} — discover available tools.

        Returns [] on network failure for graceful degradation.
        """
        await self.ensure_agent()
        try:
            resp = await self._get_client().get(f"/tools/{self.agent_id}")
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.warning("Tool discovery unavailable — returning empty list")
            return []
        self._handle_response(resp, "get tools")
        data = resp.json()
        tools = data.get("tools", []) if isinstance(data, dict) else data
        if format == "openai":
            return [{k: v for k, v in t.items() if k != "metadata"} for t in tools]
        return tools

    async def call_tool(self, tool_name: str, input_data: dict | None = None) -> dict:
        """POST /tools/{agent_id}/execute — execute a marketplace tool."""
        await self.ensure_agent()
        resp = await self._get_client().post(
            f"/tools/{self.agent_id}/execute",
            json={"tool_name": tool_name, "input": input_data or {}},
        )
        code = resp.status_code
        if code == 428:
            raise KateCredentialError(
                f"[KATE] Tool '{tool_name}' requires credentials not yet configured (status 428)"
            )
        if code == 402:
            raise KateBalanceError(
                f"[KATE] Insufficient token balance to execute tool '{tool_name}' (status 402)"
            )
        if code in (502, 504):
            raise KateToolError(
                f"[KATE] Tool '{tool_name}' execution failed (status {code})"
            )
        self._handle_response(resp, f"execute tool '{tool_name}'")
        return resp.json()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
