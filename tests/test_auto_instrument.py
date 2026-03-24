"""Tests for SDK auto-instrumentation (all mocked, no real LLM calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_otel_span(
    *,
    name: str = "ChatCompletion",
    span_kind: str = "LLM",
    input_value: str = "Hello",
    output_value: str = "Hi there",
    model_name: str | None = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    start_time: int = 1_000_000_000,  # 1s in ns
    end_time: int = 2_000_000_000,  # 2s in ns
    error: bool = False,
) -> MagicMock:
    """Create a mock OTel ReadableSpan with openinference attributes."""
    span = MagicMock()
    span.name = name
    span.start_time = start_time
    span.end_time = end_time

    attrs = {
        "openinference.span.kind": span_kind,
        "input.value": input_value,
        "output.value": output_value,
        "llm.token_count.prompt": prompt_tokens,
        "llm.token_count.completion": completion_tokens,
    }
    if model_name is not None:
        attrs["llm.model_name"] = model_name
    span.attributes = attrs

    status = MagicMock()
    if error:
        status.status_code.name = "ERROR"
        status.description = "Something went wrong"
    else:
        status.status_code.name = "OK"
        status.description = None
    span.status = status

    return span


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_detect_installed_finds_openai():
    """When openai and its instrumentor are importable, detect_installed returns the entry."""
    from projectkate.instrument.registry import detect_installed

    with patch("importlib.import_module") as mock_import:
        # Both framework and instrumentor succeed
        mock_import.return_value = MagicMock()
        result = detect_installed()

    assert any(e["name"] == "openai" for e in result)


def test_detect_installed_skips_missing():
    """When a framework is not installed, it's skipped."""
    from projectkate.instrument.registry import detect_installed

    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError("not found")
        result = detect_installed()

    assert result == []


def test_detect_installed_skips_missing_instrumentor():
    """Framework installed but instrumentor missing → skipped."""
    from projectkate.instrument.registry import detect_installed

    def selective_import(name):
        # Framework modules succeed, instrumentor modules fail
        if "openinference" in name:
            raise ImportError("instrumentor not installed")
        return MagicMock()

    with patch("importlib.import_module", side_effect=selective_import):
        result = detect_installed()

    assert result == []


# ---------------------------------------------------------------------------
# Exporter tests
# ---------------------------------------------------------------------------

def test_exporter_converts_llm_span():
    """OTel LLM span is converted to SpanRecord with correct fields."""
    from projectkate._state import KateSDK
    from projectkate.instrument.exporter import KateSpanExporter

    sdk = KateSDK()
    exporter = KateSpanExporter(sdk)
    span = _make_mock_otel_span()

    record = exporter._otel_to_span_record(span)

    assert record is not None
    assert record.name == "ChatCompletion"
    assert record.input == "Hello"
    assert record.output == "Hi there"
    assert record.span_kind == "LLM"
    assert record.duration_ms == pytest.approx(1000.0)


def test_exporter_skips_internal_spans():
    """Spans without a tracked openinference.span.kind are filtered out."""
    from projectkate._state import KateSDK
    from projectkate.instrument.exporter import KateSpanExporter

    sdk = KateSDK()
    exporter = KateSpanExporter(sdk)
    span = _make_mock_otel_span(span_kind="INTERNAL")

    record = exporter._otel_to_span_record(span)
    assert record is None


def test_exporter_records_to_active_context():
    """Exported spans land in the active RunContext via sdk.record_span()."""
    from projectkate._state import KateSDK
    from projectkate.context import RunContext
    from projectkate.instrument.exporter import KateSpanExporter, SpanExportResult

    sdk = KateSDK()
    ctx = RunContext()
    sdk._active_ctx = ctx

    exporter = KateSpanExporter(sdk)
    span = _make_mock_otel_span()
    result = exporter.export([span])

    assert result == SpanExportResult.SUCCESS
    assert len(ctx.spans) == 1
    assert ctx.spans[0].name == "ChatCompletion"


def test_model_and_tokens_extracted():
    """Model name and token counts are extracted from OTel attributes."""
    from projectkate._state import KateSDK
    from projectkate.instrument.exporter import KateSpanExporter

    sdk = KateSDK()
    exporter = KateSpanExporter(sdk)
    span = _make_mock_otel_span(model_name="gpt-4o", prompt_tokens=50, completion_tokens=100)

    record = exporter._otel_to_span_record(span)

    assert record is not None
    assert record.model == "gpt-4o"
    assert record.token_count == 150


# ---------------------------------------------------------------------------
# Setup + init tests
# ---------------------------------------------------------------------------

def test_setup_instruments_detected_frameworks():
    """setup_auto_instrumentation calls .instrument() on detected frameworks."""
    from projectkate._state import KateSDK

    sdk = KateSDK()
    mock_instrumentor = MagicMock()

    with (
        patch("projectkate.instrument.registry.detect_installed") as mock_detect,
        patch("projectkate.instrument.registry.load_instrumentor") as mock_load,
        patch("opentelemetry.sdk.trace.TracerProvider") as mock_tp_cls,
        patch("opentelemetry.sdk.trace.export.SimpleSpanProcessor"),
    ):
        mock_detect.return_value = [{"name": "openai", "instrumentor": "...", "class": "..."}]
        mock_load.return_value = mock_instrumentor
        mock_tp_cls.return_value = MagicMock()

        from projectkate.instrument import setup_auto_instrumentation

        setup_auto_instrumentation(sdk)

    mock_instrumentor.instrument.assert_called_once()
    assert sdk._tracer_provider is not None


def test_init_auto_instrument_flag():
    """kate.init(auto_instrument=True) activates instrumentation."""
    from projectkate._state import KateSDK

    sdk = KateSDK()

    with patch("projectkate.instrument.setup_auto_instrumentation") as mock_setup:
        sdk.init(api_url="http://localhost:8000", agent_id="test", auto_instrument=True)

    mock_setup.assert_called_once_with(sdk)


def test_init_without_auto_instrument():
    """Default init() does NOT activate instrumentation."""
    from projectkate._state import KateSDK

    sdk = KateSDK()

    with patch("projectkate.instrument.setup_auto_instrumentation") as mock_setup:
        sdk.init(api_url="http://localhost:8000", agent_id="test")

    mock_setup.assert_not_called()


def test_reset_cleans_up_provider():
    """sdk.reset() shuts down the tracer provider."""
    from projectkate._state import KateSDK

    sdk = KateSDK()
    KateSDK._instance = sdk

    mock_provider = MagicMock()
    sdk._tracer_provider = mock_provider

    KateSDK.reset()

    mock_provider.shutdown.assert_called_once()
    assert KateSDK._instance is None
