"""kate.run() async context manager."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from projectkate._state import KateSDK
from projectkate.constants import DEFAULT_TRIGGER, VALID_TRIGGERS
from projectkate.context import RunContext
from projectkate.output import print_eval_summary


@asynccontextmanager
async def run(*, trigger: str = DEFAULT_TRIGGER) -> AsyncIterator[RunContext]:
    """Async context manager that captures spans and runs eval on exit.

    Usage::

        async with kate.run():
            result = summarize(email_body)
        # ← Evals run here
    """
    if trigger not in VALID_TRIGGERS:
        raise ValueError(
            f"Invalid trigger: {trigger!r}. Must be one of: {sorted(VALID_TRIGGERS)}"
        )

    sdk = KateSDK.get()
    ctx = RunContext()
    sdk._active_ctx = ctx

    try:
        if sdk.mode == "remote" and sdk._remote_runner:
            data = await sdk._remote_runner.start_run(ctx.run_id, trigger)
            ctx.run_id = str(data.get("id", ctx.run_id))

        yield ctx
    finally:
        sdk._active_ctx = None
        summary = await sdk.finish_run(ctx)
        if summary:
            print_eval_summary(summary)
