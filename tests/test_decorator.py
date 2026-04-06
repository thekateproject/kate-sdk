"""Tests for @kate.trace() decorator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import projectkate as kate
from projectkate._state import KateSDK
from projectkate.context import RunContext
from projectkate.extensions import Extension


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


# -- Extension integration tests --


def _make_remote_sdk():
    """Set up SDK in remote mode with a mock runner."""
    kate.init(
        api_url="https://kate.test:8000",
        api_key="test-key",
        agent_id="00000000-0000-0000-0000-000000000001",
    )
    sdk = KateSDK.get()
    ctx = RunContext()
    sdk._active_ctx = ctx
    return sdk, ctx


_ACTIVE_EXT = Extension(
    id="ext-00000000-0000-0000-0000-000000000099",
    target_function="enhance_me",
    status="active",
    version=1,
)


@pytest.mark.asyncio
async def test_trace_async_with_extension():
    """Remote mode + active extension + odd call → enhanced output returned."""
    sdk, ctx = _make_remote_sdk()

    # Reset the A/B counter so the second call is enhanced (odd = 1)
    import projectkate.decorators as dec
    dec._ab_counters.clear()
    dec._extension_cache.clear()

    @kate.trace("enhance_me")
    async def enhance_me(text):
        return f"original: {text}"

    with patch.object(dec, "_get_cached_extension", new_callable=AsyncMock) as mock_ext, \
         patch.object(dec, "_execute_extension", new_callable=AsyncMock) as mock_exec, \
         patch.object(dec, "_report_ab", new_callable=AsyncMock) as mock_report:
        mock_ext.return_value = _ACTIVE_EXT
        mock_exec.return_value = {"success": True, "output": "enhanced: hello"}

        # First call (counter=0, even → original, no execute)
        r1 = await enhance_me("hello")
        assert r1 == "original: hello"

        # Second call (counter=1, odd → enhanced)
        r2 = await enhance_me("hello")
        assert r2 == "enhanced: hello"
        mock_exec.assert_called_once()
        mock_report.assert_called_once_with(sdk, _ACTIVE_EXT.id, enhanced_won=True)

    assert len(ctx.spans) == 2
    assert ctx.spans[1].enhanced is True
    assert ctx.spans[1].extension_id == _ACTIVE_EXT.id


@pytest.mark.asyncio
async def test_trace_async_extension_fallback():
    """Extension execute fails → original output returned."""
    sdk, ctx = _make_remote_sdk()

    import projectkate.decorators as dec
    dec._ab_counters.clear()
    dec._extension_cache.clear()
    # Pre-set counter to 1 so next call is odd (enhanced path)
    dec._ab_counters[_ACTIVE_EXT.id] = 1

    @kate.trace("enhance_me")
    async def enhance_me(text):
        return f"original: {text}"

    with patch.object(dec, "_get_cached_extension", new_callable=AsyncMock) as mock_ext, \
         patch.object(dec, "_execute_extension", new_callable=AsyncMock) as mock_exec, \
         patch.object(dec, "_report_ab", new_callable=AsyncMock) as mock_report:
        mock_ext.return_value = _ACTIVE_EXT
        mock_exec.return_value = {"success": False, "error": "compile error"}

        result = await enhance_me("hello")
        assert result == "original: hello"  # falls back
        mock_report.assert_called_once_with(sdk, _ACTIVE_EXT.id, enhanced_won=False)

    assert ctx.spans[0].enhanced is False


@pytest.mark.asyncio
async def test_trace_async_no_extension():
    """Remote mode but no active extension → original output."""
    sdk, ctx = _make_remote_sdk()

    import projectkate.decorators as dec
    dec._extension_cache.clear()

    @kate.trace("no_ext_fn")
    async def no_ext_fn():
        return "plain"

    with patch.object(dec, "_get_cached_extension", new_callable=AsyncMock) as mock_ext:
        mock_ext.return_value = None
        result = await no_ext_fn()

    assert result == "plain"
    assert ctx.spans[0].enhanced is False
    assert ctx.spans[0].extension_id is None


def test_trace_sync_ignores_extensions():
    """Sync function → no extension check (extensions are async-only)."""
    sdk, ctx = _make_remote_sdk()

    import projectkate.decorators as dec
    dec._extension_cache.clear()

    @kate.trace("sync_fn")
    def sync_fn():
        return "sync result"

    with patch.object(dec, "_get_cached_extension", new_callable=AsyncMock) as mock_ext:
        result = sync_fn()

    assert result == "sync result"
    mock_ext.assert_not_called()
    assert len(ctx.spans) == 1
