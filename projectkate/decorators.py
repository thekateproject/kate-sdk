"""@kate.trace() decorator — captures sync and async function spans."""

from __future__ import annotations

import asyncio
import functools
import json
import time
from typing import Any, Callable

from projectkate._state import KateSDK
from projectkate.context import SpanRecord


_REDACT_KEYS = frozenset({"api_key", "password", "token", "secret", "credentials"})

# Extension caching and A/B state
_extension_cache: dict[str, tuple[Any, float]] = {}
_CACHE_TTL = 300  # 5 minutes
_ab_counters: dict[str, int] = {}


def _redact_kwargs(kwargs: dict) -> dict:
    return {k: "[REDACTED]" if k in _REDACT_KEYS else v for k, v in kwargs.items()}


def _serialize(value: Any) -> str:
    """Best-effort serialization of function args/return values."""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def trace(name: str | None = None, *, span_kind: str = "LLM", kind: str | None = None) -> Callable:
    """Decorator factory that records a span for each function call.

    Usage::

        @kate.trace("summarize_email")
        def summarize(text: str) -> str:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__
        docstring = (fn.__doc__ or "").strip() or None

        def _record(sdk: KateSDK, input_str: str, output: Any, error: str | None,
                     duration_ms: float, *, enhanced: bool = False,
                     extension_id: str | None = None) -> None:
            sdk.record_span(
                SpanRecord(
                    name=span_name,
                    input=input_str,
                    output=_serialize(output) if output is not None else "",
                    span_kind=span_kind,
                    duration_ms=duration_ms,
                    error=error,
                    enhanced=enhanced,
                    extension_id=extension_id,
                    kind=kind,
                    docstring=docstring,
                )
            )

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                sdk = KateSDK.get()
                input_str = _serialize({"args": args, "kwargs": _redact_kwargs(kwargs)})
                start = time.perf_counter()
                error = None
                output = None
                enhanced = False
                extension_id = None
                try:
                    output = await fn(*args, **kwargs)

                    # Extension hook (remote mode only)
                    if sdk.mode == "remote" and sdk._remote_runner:
                        try:
                            ext = await _get_cached_extension(sdk, span_name)
                            if ext and ext.status == "active":
                                extension_id = ext.id
                                if _should_use_enhanced(ext.id):
                                    result = await _execute_extension(
                                        sdk, span_name,
                                        {"original_input": input_str,
                                         "original_output": output},
                                    )
                                    if result and result.get("success"):
                                        enhanced = True
                                        raw_output = result.get("output", output)
                                        if isinstance(raw_output, dict) and "enhanced_output" in raw_output:
                                            output = raw_output["enhanced_output"]
                                        else:
                                            output = raw_output
                                        await _report_ab(sdk, ext.id, enhanced_won=True)
                                    else:
                                        await _report_ab(sdk, ext.id, enhanced_won=False)
                        except Exception:
                            pass  # Never break original function

                    return output
                except Exception as exc:
                    error = str(exc)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    try:
                        _record(sdk, input_str, output, error, duration_ms,
                                enhanced=enhanced, extension_id=extension_id)
                    except Exception:
                        pass

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            sdk = KateSDK.get()
            input_str = _serialize({"args": args, "kwargs": _redact_kwargs(kwargs)})
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


async def _get_cached_extension(sdk: KateSDK, function_name: str):
    """Fetch extension metadata, cached for 5 minutes."""
    cached = _extension_cache.get(function_name)
    if cached and (time.time() - cached[1]) < _CACHE_TTL:
        return cached[0]
    runner = sdk._remote_runner
    from projectkate.extensions import Extension
    try:
        client = runner._get_client()
        resp = await client.get(f"/extensions/{runner.agent_id}/{function_name}")
        if resp.status_code == 404:
            ext = None
        elif resp.is_success:
            data = resp.json()
            ext = Extension(
                id=str(data["id"]),
                target_function=data["target_function"],
                status=data["status"],
                version=data["version"],
            )
        else:
            ext = None
    except Exception:
        ext = None
    _extension_cache[function_name] = (ext, time.time())
    return ext


def _should_use_enhanced(extension_id: str) -> bool:
    """A/B alternation: even calls → original, odd calls → enhanced."""
    counter = _ab_counters.get(extension_id, 0)
    _ab_counters[extension_id] = counter + 1
    return counter % 2 == 1


async def _execute_extension(sdk: KateSDK, function_name: str, input_data: dict):
    """Execute an extension via the API."""
    runner = sdk._remote_runner
    client = runner._get_client()
    resp = await client.post(
        f"/extensions/{runner.agent_id}/execute",
        json={"function_name": function_name, "input_data": input_data},
    )
    if resp.is_success:
        return resp.json()
    return None


async def _report_ab(sdk: KateSDK, extension_id: str, *, enhanced_won: bool):
    """Report A/B result. Never blocks on failure."""
    try:
        runner = sdk._remote_runner
        client = runner._get_client()
        await client.post(
            f"/extensions/{runner.agent_id}/ab-result",
            json={"extension_id": extension_id, "enhanced_won": enhanced_won},
        )
    except Exception:
        pass
