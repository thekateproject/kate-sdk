"""KateClient — management client for the Kate platform API."""

from __future__ import annotations

import warnings
from typing import Any, TYPE_CHECKING

import httpx

from projectkate.remote.runner import KateRemoteError

if TYPE_CHECKING:
    from projectkate.agents import AgentsClient
    from projectkate.artifacts import ArtifactsClient
    from projectkate.briefs import BriefsClient
    from projectkate.discovery import DiscoveryClient
    from projectkate.evals import EvalsClient
    from projectkate.runs import RunsClient
    from projectkate.wallet import WalletClient


class KateClient:
    """Management client for programmatic access to the Kate platform.

    Usage::

        client = KateClient(api_key="kate_abc123")
        agents = await client.agents.list()
    """

    DEFAULT_BASE_URL = "https://api.projectkate.com"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None
        if (
            not base_url.startswith("https://")
            and "localhost" not in base_url
            and "127.0.0.1" not in base_url
        ):
            warnings.warn(
                "KATE API URL uses HTTP — API keys may be transmitted in plaintext",
                stacklevel=2,
            )

        # Lazy sub-clients
        self._agents: AgentsClient | None = None
        self._evals: EvalsClient | None = None
        self._discovery: DiscoveryClient | None = None
        self._briefs: BriefsClient | None = None
        self._artifacts: ArtifactsClient | None = None
        self._wallet: WalletClient | None = None
        self._runs: RunsClient | None = None

    def __repr__(self) -> str:
        return f"KateClient(base_url={self._base_url!r})"

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"x-api-key": self._api_key},
                timeout=self._timeout,
            )
        return self._http

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        data: Any = None,
        files: Any = None,
    ) -> dict | list:
        """Send an HTTP request and return parsed JSON response."""
        client = self._get_http()
        resp = await client.request(
            method,
            path,
            json=json,
            params=params,
            data=data,
            files=files,
        )
        _handle_response(resp, f"{method} {path}")
        if resp.status_code == 204:
            return {}
        return resp.json()

    # -- Sub-client properties (lazy) --

    @property
    def agents(self) -> AgentsClient:
        if self._agents is None:
            from projectkate.agents import AgentsClient
            self._agents = AgentsClient(self)
        return self._agents

    @property
    def evals(self) -> EvalsClient:
        if self._evals is None:
            from projectkate.evals import EvalsClient
            self._evals = EvalsClient(self)
        return self._evals

    @property
    def discovery(self) -> DiscoveryClient:
        if self._discovery is None:
            from projectkate.discovery import DiscoveryClient
            self._discovery = DiscoveryClient(self)
        return self._discovery

    @property
    def briefs(self) -> BriefsClient:
        if self._briefs is None:
            from projectkate.briefs import BriefsClient
            self._briefs = BriefsClient(self)
        return self._briefs

    @property
    def artifacts(self) -> ArtifactsClient:
        if self._artifacts is None:
            from projectkate.artifacts import ArtifactsClient
            self._artifacts = ArtifactsClient(self)
        return self._artifacts

    @property
    def wallet(self) -> WalletClient:
        if self._wallet is None:
            from projectkate.wallet import WalletClient
            self._wallet = WalletClient(self)
        return self._wallet

    @property
    def runs(self) -> RunsClient:
        if self._runs is None:
            from projectkate.runs import RunsClient
            self._runs = RunsClient(self)
        return self._runs

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> KateClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def _handle_response(resp: httpx.Response, operation: str) -> None:
    """Raise KateRemoteError for non-success responses."""
    if resp.is_success:
        return
    code = resp.status_code
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text

    if code == 401:
        raise KateRemoteError(
            f"[KATE] Authentication failed during {operation}. "
            "Check that your api_key is correct."
        )
    if code == 404:
        raise KateRemoteError(
            f"[KATE] Resource not found during {operation} (404). Detail: {detail}"
        )
    if code >= 500:
        raise KateRemoteError(
            f"[KATE] {operation} failed with status {code}. Internal server error"
        )
    raise KateRemoteError(
        f"[KATE] {operation} failed with status {code}. Detail: {detail}"
    )
