"""KateSDK singleton — mode routing and global state."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from projectkate.constants import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    KNOWN_LLM_PROVIDERS,
)

if TYPE_CHECKING:
    from projectkate._llm import LLMClient
    from projectkate.context import RunContext, SpanRecord
    from projectkate.remote.runner import RemoteEvalRunner

logger = logging.getLogger(__name__)


_lock = threading.Lock()


class KateSDK:
    """Singleton holding SDK configuration and active run context."""

    _instance: KateSDK | None = None

    def __init__(self) -> None:
        self.mode: str = "local"  # "local" or "remote"
        self._active_ctx: RunContext | None = None
        self._llm_client: LLMClient | None = None
        self._remote_runner: RemoteEvalRunner | None = None
        self._store_results: bool = True
        self._tracer_provider = None

    @classmethod
    def get(cls) -> KateSDK:
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        if cls._instance is not None:
            if cls._instance._tracer_provider is not None:
                try:
                    cls._instance._tracer_provider.shutdown()
                except Exception:
                    logger.debug("Failed to shut down tracer provider", exc_info=True)
            if cls._instance._remote_runner is not None:
                try:
                    client = cls._instance._remote_runner._client
                    if client is not None:
                        cls._instance._remote_runner._client = None
                except Exception:
                    logger.debug("Failed to close remote runner client", exc_info=True)
        cls._instance = None

    def init(
        self,
        *,
        llm_api_key: str | None = None,
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        agent_id: str | None = None,
        agent_name: str | None = None,
        agent_objective: str | None = None,
        agent_domain: str | None = None,
        store_results: bool = True,
        auto_instrument: bool = False,
    ) -> None:
        """Configure the SDK. Auto-detects local vs remote mode."""
        api_url = api_url or os.environ.get("KATE_API_URL")
        api_key = api_key or os.environ.get("KATE_API_KEY")
        agent_id = agent_id or os.environ.get("KATE_AGENT_ID")
        agent_domain = agent_domain or os.environ.get("KATE_AGENT_DOMAIN")
        self._store_results = store_results

        if api_url:
            # Remote mode
            self.mode = "remote"
            if not api_key:
                raise ValueError(
                    "api_key is required when api_url is set (remote mode)"
                )
            if not agent_id and not agent_name:
                raise ValueError(
                    "Remote mode requires either agent_id= or agent_name=. "
                    "Pass one to projectkate.init() or set KATE_AGENT_ID."
                )
            from projectkate.remote.runner import RemoteEvalRunner

            self._remote_runner = RemoteEvalRunner(
                api_url=api_url,
                api_key=api_key or "",
                agent_id=agent_id or "",
                agent_name=agent_name,
                agent_objective=agent_objective,
                agent_domain=agent_domain,
            )
        else:
            # Local mode — needs an LLM key for the eval judge
            self.mode = "local"
            llm_api_key = llm_api_key or os.environ.get("KATE_LLM_API_KEY")
            if not llm_api_key:
                raise ValueError(
                    "Local mode requires an LLM API key. "
                    "Pass llm_api_key= or set KATE_LLM_API_KEY."
                )

            if llm_provider not in KNOWN_LLM_PROVIDERS:
                raise ValueError(
                    f"Unsupported llm_provider: {llm_provider!r}. "
                    f"Must be one of: {sorted(KNOWN_LLM_PROVIDERS)}"
                )

            if llm_provider == "openai":
                from projectkate._llm import OpenAILLMClient

                self._llm_client = OpenAILLMClient(
                    api_key=llm_api_key, model=llm_model or DEFAULT_OPENAI_MODEL
                )
            else:
                from projectkate._llm import AnthropicLLMClient

                self._llm_client = AnthropicLLMClient(
                    api_key=llm_api_key, model=llm_model or DEFAULT_ANTHROPIC_MODEL
                )

        if auto_instrument:
            from projectkate.instrument import setup_auto_instrumentation

            setup_auto_instrumentation(self)

    def record_span(self, span: SpanRecord) -> None:
        if self._active_ctx is not None:
            self._active_ctx.add_span(span)

    async def finish_run(self, ctx: RunContext) -> dict | None:
        """Route to local or remote runner. Returns eval summary dict."""
        if self.mode == "remote" and self._remote_runner:
            await self._remote_runner.upload_spans(ctx.run_id, ctx.spans)
            await self._remote_runner.complete_run(ctx.run_id)
            return {"mode": "remote", "run_id": ctx.run_id, "status": "submitted"}

        # Local mode
        from projectkate.local.runner import LocalEvalRunner

        runner = LocalEvalRunner(
            llm_client=self._llm_client,
            store_results=self._store_results,
        )
        return await runner.run(ctx.spans)

    async def get_tools(self, format: str = "openai") -> list[dict]:
        """Discover available marketplace tools for the current agent."""
        if not self._remote_runner:
            raise RuntimeError(
                "get_tools() requires remote mode. "
                "Call projectkate.init(api_url=..., api_key=...) first."
            )
        return await self._remote_runner.get_tools(format=format)

    async def call_tool(self, tool_name: str, input_data: dict | None = None) -> dict:
        """Execute a marketplace tool. Returns raw response dict."""
        if not self._remote_runner:
            raise RuntimeError(
                "call_tool() requires remote mode. "
                "Call projectkate.init(api_url=..., api_key=...) first."
            )
        return await self._remote_runner.call_tool(tool_name, input_data)

    async def poll_run_status(
        self, run_id: str, *, interval_seconds: float = 2.0, timeout_seconds: float = 300.0,
    ) -> dict:
        """Poll remote run until terminal status. Delegates to remote runner."""
        if not self._remote_runner:
            raise RuntimeError("poll_run_status requires remote mode")
        return await self._remote_runner.poll_run_status(
            run_id, interval_seconds=interval_seconds, timeout_seconds=timeout_seconds,
        )
