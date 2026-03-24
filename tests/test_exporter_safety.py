"""Tests for OTel exporter null safety."""

from __future__ import annotations

from unittest.mock import MagicMock

from projectkate._state import KateSDK
from projectkate.instrument.exporter import KateSpanExporter


def _make_span(*, status=None, status_code=None, attrs=None):
    """Create a mock OTel ReadableSpan."""
    span = MagicMock()
    span.name = "test_span"
    span.attributes = attrs or {
        "openinference.span.kind": "LLM",
        "input.value": "hello",
        "output.value": "world",
    }
    span.start_time = 1000000000
    span.end_time = 2000000000
    span.status = status
    if status is not None and status_code is not None:
        span.status.status_code = status_code
    return span


def test_null_span_status():
    """Span with status=None should not crash."""
    sdk = KateSDK.get()
    exporter = KateSpanExporter(sdk)

    span = _make_span(status=None)
    record = exporter._otel_to_span_record(span)

    assert record is not None
    assert record.error is None


def test_null_status_code():
    """Span with status but status_code=None should not crash."""
    sdk = KateSDK.get()
    exporter = KateSpanExporter(sdk)

    status = MagicMock()
    status.status_code = None
    span = _make_span(status=status, status_code=None)
    # Override since _make_span sets status_code on status
    span.status = status

    record = exporter._otel_to_span_record(span)

    assert record is not None
    assert record.error is None


def test_error_status_captured():
    """Span with ERROR status should capture the description."""
    sdk = KateSDK.get()
    exporter = KateSpanExporter(sdk)

    status_code = MagicMock()
    status_code.name = "ERROR"
    status = MagicMock()
    status.status_code = status_code
    status.description = "something went wrong"

    span = _make_span(status=status, status_code=status_code)

    record = exporter._otel_to_span_record(span)

    assert record is not None
    assert record.error == "something went wrong"
