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

_APP_HELP = (
    "Detecta drift estadístico entre dos datasets y reporta columnas que\n"
    "han cambiado de distribución entre entrenamiento y producción.\n\n"
    "Formatos soportados: .parquet  .csv  .ndjson\n"
    "Se pueden mezclar formatos (ej: referencia .csv y producción .parquet).\n\n"
    "Exit codes:\n"
    "  0  Sin drift relevante (o --fail-on never)\n"
    "  1  Drift detectado al nivel configurado (--fail-on)\n"
    "  2  Error de entrada: formato no soportado u otro problema\n"
)

# Cada ejemplo como párrafo separado (\n\n) para que Typer+Rich los renderice en líneas distintas.
_COMPARE_EPILOG = (
    "Ejemplos:\n\n"
    "  $ dqa compare train.parquet prod.parquet\n\n"
    "  $ dqa compare train.parquet prod.parquet -c age,price\n\n"
    "  $ dqa compare train.parquet prod.parquet -f markdown -o reporte.md\n\n"
    "  $ dqa compare train.parquet prod.parquet --fail-on warning\n\n"
    "  $ dqa compare ref.csv prod.parquet\n"
)

app = typer.Typer(
    name="dqa",
    help=_APP_HELP,
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


@app.command(epilog=_COMPARE_EPILOG)
def compare(
    reference: str = typer.Argument(
        ...,
        help=(
            "Ruta al dataset de referencia (entrenamiento). "
            "Formatos: .parquet, .csv, .ndjson"
        ),
    ),
    production: str = typer.Argument(
        ...,
        help=(
            "Ruta al dataset de producción. "
            "Puede ser un formato distinto al de referencia."
        ),
    ),
    columns: Optional[str] = typer.Option(
        None,
        "--columns",
        "-c",
        help=(
            "Columnas numéricas a analizar, separadas por coma (ej: age,price). "
            "Por defecto analiza todas las columnas numéricas comunes a ambos datasets."
        ),
    ),
    format: OutputFormat = typer.Option(
        "terminal",
        "--format",
        "-f",
        help=(
            "Formato del reporte de salida. "
            "'terminal' muestra una tabla coloreada en consola; "
            "'markdown' genera texto Markdown (usar con --output para guardar en archivo)."
        ),
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Ruta del archivo donde guardar el reporte (solo con --format markdown). "
            "Si se omite, el reporte Markdown se imprime en la terminal."
        ),
    ),
    fail_on: FailOn = typer.Option(
        "alert",
        "--fail-on",
        help=(
            "Nivel mínimo de drift que causa exit code 1 (util para CI/CD). "
            "'alert': solo falla en drift severo (PSI > 0.2). "
            "'warning': falla ante cualquier drift moderado (PSI > 0.1). "
            "'never': siempre retorna 0 sin importar el resultado."
        ),
    ),
    bayesian: bool = typer.Option(
        False,
        "--bayesian",
        help=(
            "Activa analisis bayesiano (Layer 3) con PyMC: ajusta Normal(mu, sigma) "
            "a cada dataset y compara intervalos HDI al 94%. "
            "Requiere dependencias opcionales: uv sync --extra bayesian"
        ),
    ),
) -> None:
    """
    Compara dos datasets y reporta drift estadistico por columna.

    Ejecuta tres capas de analisis sobre cada columna numerica comun:
    KS test + PSI (Layer 1) y divergencia KL/JS (Layer 2).
    Opcionalmente, analisis bayesiano con PyMC (Layer 3, requiere --bayesian).
    """
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
