"""Span recording and run context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class SpanRecord:
    name: str
    input: str
    output: str
    span_kind: str = "LLM"
    duration_ms: float = 0.0
    error: str | None = None
    model: str | None = None
    token_count: int | None = None
    enhanced: bool = False
    extension_id: str | None = None


@dataclass
class RunContext:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spans: list[SpanRecord] = field(default_factory=list)

    def add_span(self, span: SpanRecord) -> None:
        self.spans.append(span)
