"""Shared test fixtures."""

from __future__ import annotations

import pytest

from kate_sdk._state import KateSDK


@pytest.fixture(autouse=True)
def reset_sdk():
    """Reset SDK singleton between tests."""
    KateSDK.reset()
    yield
    KateSDK.reset()
