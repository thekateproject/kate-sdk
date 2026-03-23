"""@kate.trace() decorator — captures sync and async function spans."""

from __future__ import annotations

import asyncio
import functools
import json
import time
from typing import Any, Callable

from kate_sdk._state import KateSDK
from kate_sdk.context import SpanRecord


def _serialize(value: Any) -> str:
    """Best-effort serialization of function args/return values."""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def trace(name: str | None = None, *, span_kind: str = "LLM") -> Callable:
    """Decorator factory that records a span for each function call.

    Usage::

        @kate.trace("summarize_email")
        def summarize(text: str) -> str:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        def _record(sdk: KateSDK, input_str: str, output: Any, error: str | None,
                     duration_ms: float) -> None:
            sdk.record_span(
                SpanRecord(
                    name=span_name,
                    input=input_str,
                    output=_serialize(output) if output is not None else "",
                    span_kind=span_kind,
                    duration_ms=duration_ms,
                    error=error,
                )
            )

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                sdk = KateSDK.get()
                input_str = _serialize({"args": args, "kwargs": kwargs})
                start = time.perf_counter()
                error = None
                output = None
                try:
                    output = await fn(*args, **kwargs)
                    return output
                except Exception as exc:
                    error = str(exc)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    try:
                        _record(sdk, input_str, output, error, duration_ms)
                    except Exception:
                        pass

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            sdk = KateSDK.get()
            input_str = _serialize({"args": args, "kwargs": kwargs})
            start = time.perf_counter()
            error = None
            output = None
            try:
                output = fn(*args, **kwargs)
                return output
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                try:
                    _record(sdk, input_str, output, error, duration_ms)
                except Exception:
                    pass

        return sync_wrapper

    return decorator
