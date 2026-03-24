"""Shared test fixtures."""

from __future__ import annotations

import pytest

from projectkate._state import KateSDK


@pytest.fixture(autouse=True)
def reset_sdk():
    """Reset SDK singleton between tests."""
    KateSDK.reset()
    yield
    KateSDK.reset()
