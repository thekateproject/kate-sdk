"""LocalEvalRunner — classify, eval, store results locally."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kate_sdk.local.classifier import NodeInfo, classify_node
from kate_sdk.local.criteria import generate_criteria
from kate_sdk.local.metrics import (
    get_metric_names_for_classification,
    get_metrics_for_classification,
    run_metric,
)

if TYPE_CHECKING:
    from kate_sdk._llm import LLMClient
    from kate_sdk.context import SpanRecord


class LocalEvalRunner:
    def __init__(self, llm_client: LLMClient | None, store_results: bool = True):
        self._llm = llm_client
        self._store = store_results

    async def run(self, spans: list[SpanRecord]) -> dict:
        """Run local eval pipeline. Returns summary dict for output."""
        results: list[dict] = []
        scores: list[float] = []
        started_at = datetime.now(UTC)

        for span in spans:
            if span.span_kind.upper() != "LLM":
                continue

            # Build NodeInfo from span — use input as prompt template
            node = NodeInfo(
                name=span.name,
                prompt_template=span.input,
                input_keys=[],
                output_keys=[],
            )

            classification_result = await classify_node(node, self._llm)
            metric_names = get_metric_names_for_classification(
                classification_result.classification,
            )

            if not metric_names:
                continue  # TRANSFORMATION — no metrics

            # Generate criteria for GEval
            criteria = None
            if "GEval" in metric_names and span.input:
                criteria = await generate_criteria(
                    node_name=span.name,
                    node_prompt=span.input,
                    classification=classification_result.classification,
                    llm_client=self._llm,
                )

            metrics = get_metrics_for_classification(
                classification_result.classification,
                criteria=criteria,
            )

            for metric in metrics:
                score_val, passed, reason = await run_metric(
                    metric, span.input, span.output, [],
                )
                result_entry = {
                    "node_name": span.name,
                    "metric_name": type(metric).__name__,
                    "score": score_val,
                    "threshold": getattr(metric, "threshold", 0.5),
                    "passed": passed,
                    "reason": reason,
                }
                results.append(result_entry)
                scores.append(score_val)

        overall_score = sum(scores) / len(scores) if scores else None
        completed_at = datetime.now(UTC)

        if self._store and results:
            await self._persist(results, overall_score, started_at, completed_at)

        return {
            "mode": "local",
            "results": results,
            "overall_score": overall_score,
        }

    async def _persist(
        self,
        results: list[dict],
        overall_score: float | None,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        """Write results to ~/.kate/evals.db."""
        from kate_sdk.local.engine import get_session
        from kate_sdk.local.models import LocalEvalResult, LocalEvalRun

        async with await get_session() as session:
            run = LocalEvalRun(
                status="completed",
                overall_score=overall_score,
                started_at=started_at,
                completed_at=completed_at,
            )
            session.add(run)
            await session.flush()

            for r in results:
                session.add(
                    LocalEvalResult(
                        run_id=run.id,
                        node_name=r["node_name"],
                        metric_name=r["metric_name"],
                        score=r["score"],
                        threshold=r["threshold"],
                        passed=r["passed"],
                        reason=r.get("reason"),
                    )
                )
            await session.commit()
