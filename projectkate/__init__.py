"""KATE SDK — local-first auto-eval for AI agents."""

import json
import logging
import time
from typing import Any, Callable

from projectkate._state import KateSDK
from projectkate.client import KateClient
from projectkate.context import SpanRecord
from projectkate.decorators import trace
from projectkate.models import ToolResult
from projectkate.remote.runner import (
    KateBalanceError,
    KateCredentialError,
    KateRemoteError,
    KateToolError,
)
from projectkate.run_context import run

logger = logging.getLogger(__name__)


def init(**kwargs) -> None:
    """Initialize the KATE SDK. See KateSDK.init() for parameters."""
    KateSDK.get().init(**kwargs)


async def poll_run_status(
    run_id: str, *, interval_seconds: float = 2.0, timeout_seconds: float = 300.0,
) -> dict:
    """Poll a remote eval run until it reaches a terminal status."""
    return await KateSDK.get().poll_run_status(
        run_id, interval_seconds=interval_seconds, timeout_seconds=timeout_seconds,
    )


async def get_tools(format: str = "openai") -> list[dict]:
    """Discover available marketplace tools for the current agent."""
    return await KateSDK.get().get_tools(format=format)


async def call_tool(tool_name: str, input_data: dict | None = None) -> ToolResult:
    """Execute a marketplace tool with tracing."""
    sdk = KateSDK.get()
    start = time.perf_counter()
    error = None
    raw: dict | None = None
    input_str = json.dumps({"tool_name": tool_name, "input": input_data}, default=str)
    try:
        raw = await sdk.call_tool(tool_name, input_data)
        return ToolResult(
            success=raw.get("success", True),
            output=raw.get("output"),
            error=raw.get("error"),
            execution_time_ms=raw.get("execution_time_ms", 0),
            tokens_charged=raw.get("tokens_charged", 0),
        )
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        try:
            sdk.record_span(
                SpanRecord(
                    name=tool_name,
                    span_kind="TOOL",
                    input=input_str,
                    output=json.dumps(raw, default=str) if raw else "",
                    duration_ms=duration_ms,
                    error=error,
                )
            )
        except Exception:
            logger.debug("Failed to record tool span for %s", tool_name, exc_info=True)


def make_kate_tool_executor(tools: list[dict]) -> dict[str, Callable]:
    """Build a {name: async_callable} map for LangGraph tool nodes.

    Each callable wraps call_tool() for the given tool name.
    """
    executors: dict[str, Callable] = {}
    for tool in tools:
        fname = tool.get("function", {}).get("name") or tool.get("name", "")
        if not fname:
            continue

        async def _invoke(input_data: dict | None = None, _name: str = fname) -> ToolResult:
            return await call_tool(_name, input_data)

        executors[fname] = _invoke
    return executors


__all__ = [
    "init",
    "trace",
    "run",
    "poll_run_status",
    "get_tools",
    "call_tool",
    "make_kate_tool_executor",
    "KateRemoteError",
    "KateToolError",
    "KateCredentialError",
    "KateBalanceError",
    "KateClient",
    "ToolResult",
]
