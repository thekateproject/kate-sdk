"""Auto-instrumentation for AI frameworks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kate_sdk._state import KateSDK

logger = logging.getLogger(__name__)


def setup_auto_instrumentation(sdk: KateSDK) -> None:
    """Create TracerProvider with KateSpanExporter, detect + activate instrumentors."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from kate_sdk.instrument.exporter import KateSpanExporter
    from kate_sdk.instrument.registry import detect_installed, load_instrumentor

    exporter = KateSpanExporter(sdk)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    installed = detect_installed()
    for entry in installed:
        try:
            instrumentor = load_instrumentor(entry)
            instrumentor.instrument(tracer_provider=provider)
            logger.info("Auto-instrumented: %s", entry["name"])
        except Exception:
            logger.warning("Failed to instrument: %s", entry["name"], exc_info=True)

    sdk._tracer_provider = provider
