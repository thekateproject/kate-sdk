"""KATE SDK — local-first auto-eval for AI agents."""

from kate_sdk._state import KateSDK
from kate_sdk.decorators import trace
from kate_sdk.remote.runner import KateRemoteError
from kate_sdk.run_context import run


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


__all__ = ["init", "trace", "run", "poll_run_status", "KateRemoteError"]
