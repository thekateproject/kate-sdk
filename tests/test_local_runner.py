"""Tests for LocalEvalRunner with mocked LLM and DeepEval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kate_sdk._llm import LLMResponse
from kate_sdk.context import SpanRecord
from kate_sdk.local.runner import LocalEvalRunner


def _make_llm_client(classification: str = "extraction"):
    """Create a mock LLM client that returns classification JSON."""
    client = AsyncMock()
    # First call: classify_node → returns classification JSON
    # Second call: generate_criteria → returns criteria text
    client.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                text=(
                    f'{{"classification": "{classification}", '
                    f'"confidence": 0.9, "reasoning": "test"}}'
                ),
                model="mock",
                usage={"input_tokens": 10, "output_tokens": 10},
            ),
            LLMResponse(
                text="Evaluate quality of extraction",
                model="mock",
                usage={"input_tokens": 10, "output_tokens": 10},
            ),
        ]
    )
    return client


def _mock_metric(score=0.85, reason="Good"):
    m = MagicMock()
    m.threshold = 0.5
    m.score = score
    m.reason = reason
    m.a_measure = AsyncMock()
    type(m).__name__ = "GEval"
    return m


@pytest.mark.asyncio
async def test_local_runner_with_mocked_metrics():
    """Full local eval with mocked DeepEval metrics."""
    llm = _make_llm_client("extraction")

    spans = [
        SpanRecord(
            name="extract_entities",
            input="Extract names from: John went to Paris with Mary",
            output='["John", "Mary", "Paris"]',
            span_kind="LLM",
            duration_ms=100.0,
        ),
    ]

    mock_m = _mock_metric(0.85, "Good extraction")

    with patch(
        "kate_sdk.local.runner.get_metrics_for_classification",
        return_value=[mock_m],
    ):
        runner = LocalEvalRunner(llm_client=llm, store_results=False)
        summary = await runner.run(spans)

    assert summary["mode"] == "local"
    assert summary["overall_score"] == 0.85
    assert len(summary["results"]) == 1
    assert summary["results"][0]["node_name"] == "extract_entities"
    assert summary["results"][0]["passed"] is True


@pytest.mark.asyncio
async def test_local_runner_empty_spans():
    """Empty spans → no results, no crash."""
    llm = AsyncMock()
    runner = LocalEvalRunner(llm_client=llm, store_results=False)
    summary = await runner.run([])

    assert summary["mode"] == "local"
    assert summary["results"] == []
    assert summary["overall_score"] is None


@pytest.mark.asyncio
async def test_local_runner_skips_non_llm_spans():
    """Non-LLM spans are skipped."""
    llm = AsyncMock()
    spans = [
        SpanRecord(name="tool_call", input="x", output="y", span_kind="TOOL"),
    ]
    runner = LocalEvalRunner(llm_client=llm, store_results=False)
    summary = await runner.run(spans)
    assert summary["results"] == []


@pytest.mark.asyncio
async def test_local_runner_accepts_lowercase_llm():
    """Lowercase 'llm' span_kind should be accepted."""
    llm = _make_llm_client("extraction")
    spans = [
        SpanRecord(
            name="extract",
            input="some input",
            output="some output",
            span_kind="llm",
            duration_ms=50.0,
        ),
    ]

    mock_m = _mock_metric(0.9, "Good")

    with patch(
        "kate_sdk.local.runner.get_metrics_for_classification",
        return_value=[mock_m],
    ):
        runner = LocalEvalRunner(llm_client=llm, store_results=False)
        summary = await runner.run(spans)

    assert len(summary["results"]) == 1
    assert summary["overall_score"] == 0.9


@pytest.mark.asyncio
async def test_local_runner_sqlite_persistence(tmp_path):
    """Results are written to SQLite when store_results=True."""
    llm = _make_llm_client("extraction")
    spans = [
        SpanRecord(
            name="extract",
            input="some input",
            output="some output",
            span_kind="LLM",
        ),
    ]

    mock_m = _mock_metric(0.9, "Good")
    db_path = tmp_path / "test_evals.db"

    with (
        patch(
            "kate_sdk.local.runner.get_metrics_for_classification",
            return_value=[mock_m],
        ),
        patch("kate_sdk.local.engine._db_path", return_value=db_path),
        patch("kate_sdk.local.engine._engine", None),
        patch("kate_sdk.local.engine._session_factory", None),
    ):
        runner = LocalEvalRunner(llm_client=llm, store_results=True)
        summary = await runner.run(spans)

    assert summary["overall_score"] == 0.9
    assert db_path.exists()
