"""
Adapter Reporter que emite el reporte de drift en la terminal usando Rich.
"""

from __future__ import annotations

from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table

from dqa.domain.models import DatasetReport, DriftLevel

_STYLE = {
    DriftLevel.STABLE: "green",
    DriftLevel.WARNING: "yellow",
    DriftLevel.ALERT: "red bold",
}
_EMOJI = {DriftLevel.STABLE: "🟢", DriftLevel.WARNING: "🟡", DriftLevel.ALERT: "🔴"}


class RichReporter:
    def __init__(self, console: Optional[Console] = None):
        self._console = console or Console()

    def report(self, result: DatasetReport) -> None:
        self._console.print(
            f"\n[bold]DQA Report[/bold] — "
            f"Overall: {_EMOJI[result.overall_level]} [{_STYLE[result.overall_level]}]{result.overall_level}[/]\n"
        )

        if result.schema_diff.has_changes:
            self._console.print("[bold yellow]⚠ Schema Drift Detected[/bold yellow]")
            for col in result.schema_diff.added:
                self._console.print(f"  ➕ Added:   [green]{col}[/green]")
            for col in result.schema_diff.removed:
                self._console.print(f"  ➖ Removed: [red]{col}[/red]")
            for col, (old, new) in result.schema_diff.type_changed.items():
                self._console.print(
                    f"  🔄 Type:    [yellow]{col}[/yellow] {old} → {new}"
                )
            self._console.print()

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("Column", style="bold")
        table.add_column("Status")
        table.add_column("Metric")
        table.add_column("Details")

        for col in result.columns:
            first_row = True
            for metric_name, analysis in col.results.items():
                details_str = "  ".join(f"{k}={v}" for k, v in analysis.details.items())
                row_style = (
                    _STYLE[analysis.level]
                    if analysis.level >= DriftLevel.WARNING
                    else None
                )
                table.add_row(
                    col.name if first_row else "",
                    f"{_EMOJI[col.worst_level]} {col.worst_level}" if first_row else "",
                    metric_name,
                    details_str,
                    style=row_style,
                )
                first_row = False

        self._console.print(table)
