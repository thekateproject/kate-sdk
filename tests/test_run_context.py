"""Tests for kate.run() context manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import kate_sdk as kate
from kate_sdk._state import KateSDK


def test_invalid_trigger_raises():
    """Invalid trigger value should raise ValueError synchronously."""
    import asyncio

    async def _run():
        async with kate.run(trigger="bad_trigger"):
            pass

    with pytest.raises(ValueError, match="Invalid trigger"):
        asyncio.run(_run())


@pytest.mark.asyncio
async def test_remote_run_calls_lifecycle_methods():
    """Remote run calls start_run, upload_spans, complete_run in order."""
    kate.init(api_url="http://localhost:8000", api_key="abc", agent_id="uuid-1")
    sdk = KateSDK.get()
    runner = sdk._remote_runner

    runner.start_run = AsyncMock(return_value={"id": "run-123"})
    runner.upload_spans = AsyncMock()
    runner.complete_run = AsyncMock()

    with patch("kate_sdk.run_context.print_eval_summary"):
        async with kate.run(trigger="manual"):
            pass

    runner.start_run.assert_called_once()
    runner.upload_spans.assert_called_once()
    runner.complete_run.assert_called_once()


@pytest.mark.asyncio
async def test_active_ctx_cleared_on_exception():
    """_active_ctx is cleared even when the run body raises."""
    kate.init(api_url="http://localhost:8000", api_key="abc", agent_id="uuid-1")
    sdk = KateSDK.get()
    runner = sdk._remote_runner

    runner.start_run = AsyncMock(return_value={"id": "run-123"})
    runner.upload_spans = AsyncMock()
    runner.complete_run = AsyncMock()

    with pytest.raises(RuntimeError, match="boom"):
        with patch("kate_sdk.run_context.print_eval_summary"):
            async with kate.run(trigger="manual"):
                raise RuntimeError("boom")

    assert sdk._active_ctx is None


@pytest.mark.asyncio
async def test_active_ctx_cleared_on_start_run_failure():
    """_active_ctx is cleared when start_run raises (no leaked context)."""
    kate.init(api_url="http://localhost:8000", api_key="abc", agent_id="uuid-1")
    sdk = KateSDK.get()
    runner = sdk._remote_runner

    runner.start_run = AsyncMock(side_effect=ConnectionError("network down"))
    runner.upload_spans = AsyncMock()
    runner.complete_run = AsyncMock()

    with pytest.raises(ConnectionError, match="network down"):
        with patch("kate_sdk.run_context.print_eval_summary"):
            async with kate.run(trigger="manual"):
                pass

    assert sdk._active_ctx is None


@pytest.mark.asyncio
async def test_run_id_updated_from_server():
    """Run ID is updated from the server response dict."""
    kate.init(api_url="http://localhost:8000", api_key="abc", agent_id="uuid-1")
    sdk = KateSDK.get()
    runner = sdk._remote_runner

    runner.start_run = AsyncMock(return_value={"id": "server-run-id"})
    runner.upload_spans = AsyncMock()
    runner.complete_run = AsyncMock()

    with patch("kate_sdk.run_context.print_eval_summary"):
        async with kate.run(trigger="manual") as ctx:
            assert ctx.run_id == "server-run-id"
