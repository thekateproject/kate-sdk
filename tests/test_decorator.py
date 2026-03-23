"""Tests for @kate.trace() decorator."""

from __future__ import annotations

import pytest

import kate_sdk as kate
from kate_sdk._state import KateSDK
from kate_sdk.context import RunContext


@pytest.fixture
def sdk_with_ctx():
    """Set up SDK with an active RunContext (no LLM needed for decorator tests)."""
    kate.init(llm_api_key="sk-test", llm_provider="anthropic")
    sdk = KateSDK.get()
    ctx = RunContext()
    sdk._active_ctx = ctx
    return sdk, ctx


def test_trace_sync(sdk_with_ctx):
    sdk, ctx = sdk_with_ctx

    @kate.trace("greet")
    def greet(name):
        return f"Hello {name}"

    result = greet("World")
    assert result == "Hello World"
    assert len(ctx.spans) == 1
    span = ctx.spans[0]
    assert span.name == "greet"
    assert "World" in span.input
    assert "Hello World" in span.output
    assert span.error is None
    assert span.duration_ms >= 0


@pytest.mark.asyncio
async def test_trace_async(sdk_with_ctx):
    sdk, ctx = sdk_with_ctx

    @kate.trace("async_greet")
    async def greet(name):
        return f"Hi {name}"

    result = await greet("Alice")
    assert result == "Hi Alice"
    assert len(ctx.spans) == 1
    assert ctx.spans[0].name == "async_greet"


def test_trace_captures_error(sdk_with_ctx):
    sdk, ctx = sdk_with_ctx

    @kate.trace("failing")
    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        fail()

    assert len(ctx.spans) == 1
    assert ctx.spans[0].error == "boom"


def test_trace_default_name(sdk_with_ctx):
    sdk, ctx = sdk_with_ctx

    @kate.trace()
    def my_func():
        return 42

    my_func()
    assert ctx.spans[0].name == "my_func"


def test_trace_no_active_context():
    """Spans are silently dropped when no RunContext is active."""
    kate.init(llm_api_key="sk-test")
    sdk = KateSDK.get()
    sdk._active_ctx = None

    @kate.trace("orphan")
    def orphan():
        return "ok"

    result = orphan()
    assert result == "ok"  # function still works
