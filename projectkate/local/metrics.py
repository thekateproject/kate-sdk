"""Metric mapper and runner for classified spans."""

from __future__ import annotations

import logging

from projectkate.local.models import NodeClassification

logger = logging.getLogger(__name__)

_CLASSIFICATION_METRICS: dict[NodeClassification, list[str]] = {
    NodeClassification.SUMMARIZATION: ["FaithfulnessMetric", "SummarizationMetric"],
    NodeClassification.RAG_QA: ["FaithfulnessMetric", "AnswerRelevancyMetric"],
    NodeClassification.SELECTION_RANKING: ["GEval"],
    NodeClassification.EXTRACTION: ["GEval"],
    NodeClassification.TRANSFORMATION: [],
}


def get_metric_names_for_classification(classification: NodeClassification) -> list[str]:
    """Return metric name strings."""
    return list(_CLASSIFICATION_METRICS[classification])


def get_metrics_for_classification(
    classification: NodeClassification,
    criteria: str | None = None,
    threshold: float = 0.5,
) -> list:
    """Return instantiated DeepEval metric objects."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        SummarizationMetric,
    )
    from deepeval.test_case import LLMTestCaseParams

    metric_map = {
        "FaithfulnessMetric": lambda: FaithfulnessMetric(threshold=threshold),
        "SummarizationMetric": lambda: SummarizationMetric(threshold=threshold),
        "AnswerRelevancyMetric": lambda: AnswerRelevancyMetric(threshold=threshold),
        "GEval": lambda: GEval(
            name="GEval",
            criteria=criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=threshold,
        ),
    }

    metric_names = _CLASSIFICATION_METRICS[classification]
    return [metric_map[name]() for name in metric_names]


async def run_metric(metric, input_text: str, output_text: str, context: list[str]):
    """Run a single DeepEval metric. Returns (score, passed, reason)."""
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=input_text,
        actual_output=output_text,
        retrieval_context=context if context else None,
    )
    try:
        await metric.a_measure(test_case)
        return (metric.score, metric.score >= metric.threshold, metric.reason)
    except Exception as exc:
        logger.warning("Metric %s failed", metric.__class__.__name__, exc_info=True)
        return (0.0, False, f"Metric error: {exc}")
