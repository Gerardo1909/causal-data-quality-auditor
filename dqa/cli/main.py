"""
Punto de entrada de la CLI. Cablea adapters y delega al engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import typer

from dqa.adapters.readers.polars_reader import PolarsReader
from dqa.adapters.reporters.markdown_reporter import MarkdownReporter
from dqa.adapters.reporters.rich_reporter import RichReporter
from dqa.domain.models import DriftLevel
from dqa.engine import run_analysis

app = typer.Typer(
    name="dqa",
    help="Detecta drift estadístico entre dos datasets.",
    add_completion=False,
)


@app.callback()
def _callback() -> None:
    pass


class OutputFormat(str, Enum):
    terminal = "terminal"
    markdown = "markdown"


class FailOn(str, Enum):
    alert = "alert"
    warning = "warning"
    never = "never"


_FAIL_THRESHOLDS: dict[FailOn, DriftLevel | None] = {
    FailOn.alert: DriftLevel.ALERT,
    FailOn.warning: DriftLevel.WARNING,
    FailOn.never: None,
}


@app.command()
def compare(
    reference: str = typer.Argument(..., help="Dataset de referencia (entrenamiento)."),
    production: str = typer.Argument(..., help="Dataset de producción."),
    columns: Optional[str] = typer.Option(
        None, "--columns", "-c", help="Columnas a analizar, separadas por coma."
    ),
    format: OutputFormat = typer.Option(
        "terminal", "--format", "-f", help="Formato de salida."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Archivo de salida (con --format markdown)."
    ),
    fail_on: FailOn = typer.Option(
        "alert", "--fail-on", help="Nivel mínimo que dispara exit code 1."
    ),
    bayesian: bool = typer.Option(
        False,
        "--bayesian",
        help="Activa análisis bayesiano (requiere uv sync --extra bayesian).",
    ),
) -> None:
    """Compara dos datasets y reporta drift estadístico por columna."""
    col_list = [c.strip() for c in columns.split(",")] if columns else None
    extra = _load_bayesian_analyzer() if bayesian else []

    try:
        report = run_analysis(
            reference,
            production,
            reader=PolarsReader(),
            columns=col_list,
            extra_analyzers=extra,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    if format == OutputFormat.terminal:
        RichReporter().report(report)
    else:
        content = MarkdownReporter().report(report)
        if output:
            with open(output, "w") as f:
                f.write(content)
            typer.echo(f"Reporte guardado en: {output}")
        else:
            typer.echo(content)

    threshold = _FAIL_THRESHOLDS[fail_on]
    if threshold is not None and report.overall_level >= threshold:
        raise typer.Exit(code=1)


def _load_bayesian_analyzer():
    """
    Importa BayesianAnalyzer en runtime para no penalizar el startup si no se usa.
    """
    try:
        from dqa.adapters.bayesian.pymc_analyzer import BayesianAnalyzer

        return [BayesianAnalyzer()]
    except ImportError:
        typer.echo(
            "⚠️  --bayesian requiere: uv sync --extra bayesian",
            err=True,
        )
        raise typer.Exit(code=2)
