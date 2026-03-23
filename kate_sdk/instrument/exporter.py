"""KateSpanExporter — converts OTel spans to SpanRecord and feeds into KATE."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from kate_sdk._state import KateSDK

logger = logging.getLogger(__name__)

# Span kinds we want to capture (skip internal/framework plumbing)
_TRACKED_SPAN_KINDS = {"LLM", "CHAIN", "TOOL", "AGENT", "EMBEDDING", "RETRIEVER"}


class KateSpanExporter(SpanExporter):
    """Lightweight OTel exporter that converts spans to SpanRecord."""

    def __init__(self, sdk: KateSDK) -> None:
        self._sdk = sdk

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            try:
                record = self._otel_to_span_record(span)
                if record is not None:
                    self._sdk.record_span(record)
            except Exception:
                logger.debug("Failed to export span %s", getattr(span, "name", "?"),
                             exc_info=True)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _otel_to_span_record(self, span: ReadableSpan):
        """Extract name, input, output, model, tokens from OTel span attributes.

        Attribute keys follow openinference semantic conventions:
        - input.value, output.value
        - llm.model_name
        - llm.token_count.prompt, llm.token_count.completion
        - openinference.span.kind
        """
        from kate_sdk.context import SpanRecord

        attrs = dict(span.attributes or {})

        span_kind = attrs.get("openinference.span.kind", "")
        if span_kind not in _TRACKED_SPAN_KINDS:
            return None

        input_val = attrs.get("input.value", "")
        output_val = attrs.get("output.value", "")

        model = attrs.get("llm.model_name")

        prompt_tokens = attrs.get("llm.token_count.prompt", 0) or 0
        completion_tokens = attrs.get("llm.token_count.completion", 0) or 0
        total_tokens = int(prompt_tokens) + int(completion_tokens)

        duration_ms = 0.0
        if span.start_time and span.end_time:
            duration_ms = (span.end_time - span.start_time) / 1_000_000  # ns -> ms

        error = None
        status_code = getattr(getattr(span, "status", None), "status_code", None)
        if status_code is not None and getattr(status_code, "name", None) == "ERROR":
            error = span.status.description

        return SpanRecord(
            name=span.name or "unknown",
            input=str(input_val)[:4000],
            output=str(output_val)[:4000],
            span_kind=str(span_kind),
            duration_ms=duration_ms,
            error=error,
            model=str(model) if model else None,
            token_count=total_tokens if total_tokens > 0 else None,
        )
