"""Rich terminal output for eval results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def print_eval_summary(summary: dict) -> None:
    """Print eval results to terminal using Rich."""
    if summary.get("mode") == "remote":
        console.print(
            f"\n[bold blue]KATE[/] Spans uploaded to remote server. "
            f"Run ID: {summary['run_id']}",
        )
        return

    results = summary.get("results", [])
    overall_score = summary.get("overall_score")

    if not results:
        console.print("\n[bold blue]KATE[/] No evaluable spans recorded.")
        return

    table = Table(title="KATE Eval Results", show_lines=True)
    table.add_column("Node", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Pass", justify="center")
    table.add_column("Reason", max_width=50)

    for r in results:
        score_str = f"{r['score']:.2f}"
        passed = "[green]PASS[/]" if r["passed"] else "[red]FAIL[/]"
        table.add_row(
            r["node_name"],
            r["metric_name"],
            score_str,
            passed,
            (r.get("reason") or "")[:80],
        )

    console.print()
    console.print(table)

    if overall_score is not None:
        color = "green" if overall_score >= 0.7 else "yellow" if overall_score >= 0.5 else "red"
        console.print(
            f"\n[bold blue]KATE[/] Overall Score: [{color}]{overall_score:.2f}[/{color}]"
        )
