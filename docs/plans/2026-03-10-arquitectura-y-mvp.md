# Causal Data Quality Auditor — Plan de Arquitectura y MVP

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** CLI liviana (`dqa compare ref.parquet prod.parquet`) que detecta drift estadístico entre dos datasets, integrable en CI/CD con exit codes correctos y sin imponer dependencias pesadas por defecto.

**Architecture:** Hexagonal (Ports & Adapters). El dominio y los algoritmos de análisis no importan ninguna librería externa — solo stdlib + numpy/scipy. Los adapters concretos (Polars para leer datos, Rich para terminal, PyMC para Bayesian) viven aislados y se inyectan en runtime. El CLI es una capa delgada que únicamente cablea adapters.

**Tech Stack:** Python 3.11+ · uv · Polars · scipy/numpy · Rich · Typer · Pydantic · PyMC (opcional, extras group)

---

## Decisiones de diseño

| Decisión | Razonamiento |
|----------|-------------|
| Ports & Adapters | El engine y el dominio no importan Polars, Rich ni PyMC. Permite testear con mocks, sustituir adapters sin tocar lógica, y mantener el core liviano para CI. |
| PyMC en extras group `[bayesian]` | PyMC con nutpie puede tardar minutos en instalarse. Un CI básico solo instala core deps y corre en segundos. |
| Solo columnas numéricas en MVP | Cubre el 80% del valor (feature drift en ML) con el menor scope. Las categóricas se agregan en v0.2 sin tocar el engine. |
| Parquet + CSV + NDJSON | Los tres formatos más comunes en pipelines de ML/data. Excel y Delta quedan en v0.2 (YAGNI). |
| Schema drift como Layer 0 | Sin deps externas, corre siempre. Detecta el caso más frecuente y más barato: columna nueva, removida o tipo cambiado. |
| `dqa/analysis/` en lugar de `dqa/tests/` | Evita colisión de nombres con el directorio `tests/` de pytest y confusión en imports. |
| Markdown report sin Jinja2 | f-strings son suficientes para Markdown. Jinja2 solo entra si se agrega HTML report (extras group `[html]`). |
| Docstrings en operaciones complejas | Cada módulo tiene docstring de una línea explicando su rol. Funciones con algoritmos no triviales documentan qué hacen, sus parámetros y qué retornan. No se documentan getters, properties simples ni código autoexplicativo. |

---

## Estructura de archivos objetivo

```
causal-data-quality-auditor/
├── dqa/
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models.py          # Dataclasses puras: ColumnReport, DatasetReport, DriftLevel
│   │   └── ports.py           # Protocols: DataReader, ColumnAnalyzer, Reporter
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── schema.py          # Layer 0: schema drift (sin deps externas)
│   │   ├── classical.py       # Layer 1: KS test, PSI (scipy)
│   │   └── information.py     # Layer 2: KL/JS Divergence (scipy/numpy)
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── readers/
│   │   │   ├── __init__.py
│   │   │   └── polars_reader.py   # Parquet, CSV, NDJSON via Polars
│   │   ├── reporters/
│   │   │   ├── __init__.py
│   │   │   ├── rich_reporter.py
│   │   │   └── markdown_reporter.py
│   │   └── bayesian/
│   │       ├── __init__.py
│   │       └── pymc_analyzer.py   # Solo se importa si --bayesian
│   ├── engine.py              # Orquesta análisis por columna; acepta adapters via DI
│   └── cli/
│       ├── __init__.py
│       └── main.py            # Typer CLI: cablea adapters, llama engine
├── tests/
│   ├── conftest.py
│   ├── domain/
│   │   └── test_models.py
│   ├── analysis/
│   │   ├── test_schema.py
│   │   ├── test_classical.py
│   │   └── test_information.py
│   ├── adapters/
│   │   ├── test_polars_reader.py
│   │   ├── test_reporters.py
│   │   └── test_bayesian.py
│   └── test_cli.py
├── docs/plans/
├── pyproject.toml
├── pytest.ini
└── README.md
```

## Branch `feat/06-reporters` — Adapters de reporte

**PR title:** `feat: reporters — Rich (terminal) y Markdown`

**Files:**
- Create: `dqa/adapters/reporters/__init__.py`
- Create: `dqa/adapters/reporters/rich_reporter.py`
- Create: `dqa/adapters/reporters/markdown_reporter.py`
- Create: `tests/adapters/test_reporters.py`

### Task 1: Tests

```python
# tests/adapters/test_reporters.py
import io
from dqa.domain.models import AnalysisResult, ColumnReport, DatasetReport, DriftLevel, SchemaDiff
from dqa.adapters.reporters.markdown_reporter import MarkdownReporter


def _make_report(level: DriftLevel) -> DatasetReport:
    return DatasetReport(
        columns=[ColumnReport(
            name="price", dtype="float64",
            results={"psi": AnalysisResult(level=level, details={"psi": 0.25})},
        )]
    )


def test_markdown_contains_column_name():
    buf = io.StringIO()
    MarkdownReporter().report(_make_report(DriftLevel.STABLE), output=buf)
    assert "price" in buf.getvalue()


def test_markdown_shows_alert_level():
    buf = io.StringIO()
    MarkdownReporter().report(_make_report(DriftLevel.ALERT), output=buf)
    assert "ALERT" in buf.getvalue()


def test_markdown_stable_does_not_show_alert():
    buf = io.StringIO()
    MarkdownReporter().report(_make_report(DriftLevel.STABLE), output=buf)
    assert "ALERT" not in buf.getvalue()


def test_markdown_shows_schema_diff():
    report = DatasetReport(
        columns=[],
        schema_diff=SchemaDiff(added=["new_col"], removed=["old_col"]),
    )
    buf = io.StringIO()
    MarkdownReporter().report(report, output=buf)
    content = buf.getvalue()
    assert "new_col" in content
    assert "old_col" in content


def test_markdown_report_returns_string():
    result = MarkdownReporter().report(_make_report(DriftLevel.STABLE))
    assert isinstance(result, str)
    assert len(result) > 0
```

### Task 2: Correr — debe fallar

```bash
uv run pytest tests/adapters/test_reporters.py -v
```

### Task 3: Implementar markdown_reporter.py

```python
# dqa/adapters/reporters/markdown_reporter.py
"""Adapter Reporter que emite el reporte de drift en formato Markdown."""
from __future__ import annotations
import io
from typing import TextIO
from dqa.domain.models import DatasetReport, DriftLevel

_EMOJI = {DriftLevel.STABLE: "🟢", DriftLevel.WARNING: "🟡", DriftLevel.ALERT: "🔴"}


class MarkdownReporter:
    """Genera reportes Markdown sin dependencias externas (no usa Jinja2)."""

    def report(self, result: DatasetReport, output: TextIO | None = None) -> str:
        """
        Genera el reporte de drift en Markdown.

        Args:
            result: DatasetReport con los resultados del análisis.
            output: Stream de escritura opcional. Si se omite, retorna el string.

        Returns:
            El contenido Markdown generado como string.
        """
        buf = output or io.StringIO()
        lines: list[str] = [
            "# DQA Drift Report\n\n",
            f"**Overall status:** {_EMOJI[result.overall_level]} {result.overall_level}\n\n",
        ]

        if result.schema_diff.has_changes:
            lines.append("## Schema Drift\n\n")
            for col in result.schema_diff.added:
                lines.append(f"- **Added:** `{col}`\n")
            for col in result.schema_diff.removed:
                lines.append(f"- **Removed:** `{col}`\n")
            for col, (old, new) in result.schema_diff.type_changed.items():
                lines.append(f"- **Type changed:** `{col}` — `{old}` → `{new}`\n")
            lines.append("\n")

        lines.append("## Column Analysis\n")
        for col in result.columns:
            emoji = _EMOJI[col.worst_level]
            lines.append(f"\n### `{col.name}` — {emoji} {col.worst_level}\n\n")
            lines.append("| Metric | Key | Value |\n|--------|-----|-------|\n")
            for metric_name, analysis in col.results.items():
                for k, v in analysis.details.items():
                    lines.append(f"| {metric_name} | {k} | {v} |\n")

        content = "".join(lines)
        buf.write(content)
        if output is None:
            return content
        return content
```

### Task 4: Implementar rich_reporter.py

```python
# dqa/adapters/reporters/rich_reporter.py
"""Adapter Reporter que emite el reporte de drift en la terminal usando Rich."""
from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich import box
from dqa.domain.models import DatasetReport, DriftLevel

_STYLE = {DriftLevel.STABLE: "green", DriftLevel.WARNING: "yellow", DriftLevel.ALERT: "red bold"}
_EMOJI = {DriftLevel.STABLE: "🟢", DriftLevel.WARNING: "🟡", DriftLevel.ALERT: "🔴"}


class RichReporter:
    def __init__(self, console: Console | None = None):
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
                self._console.print(f"  🔄 Type:    [yellow]{col}[/yellow] {old} → {new}")
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
                row_style = _STYLE[analysis.level] if analysis.level >= DriftLevel.WARNING else None
                table.add_row(
                    col.name if first_row else "",
                    f"{_EMOJI[col.worst_level]} {col.worst_level}" if first_row else "",
                    metric_name,
                    details_str,
                    style=row_style,
                )
                first_row = False

        self._console.print(table)
```

### Task 5: Correr — deben pasar

```bash
uv run pytest tests/adapters/test_reporters.py -v
# Esperado: 5 passed
```

### Task 6: Commit y push

```bash
touch dqa/adapters/reporters/__init__.py
git add dqa/adapters/reporters/ tests/adapters/test_reporters.py
git commit -m "feat: reporters — Rich (terminal) y Markdown"
git push origin feat/06-reporters
```

**PR description:**
```
## Summary
- `MarkdownReporter`: genera reportes sin Jinja2, puro f-strings
- `RichReporter`: tabla coloreada en terminal con semáforos por columna
- Ambos cubren schema diff y análisis por columna
- `MarkdownReporter` acepta `output` stream para testear sin I/O

## Test plan
- [ ] `uv run pytest tests/adapters/test_reporters.py -v` → 5 passed
- [ ] Smoke test visual: `uv run python -c "from dqa.adapters.reporters.rich_reporter import RichReporter; ..."`
```

---

## Branch `feat/07-cli` — Capa de entrada CLI

**PR title:** `feat: CLI Typer — compare con exit codes para CI/CD`

**Files:**
- Create: `dqa/cli/__init__.py`
- Create: `dqa/cli/main.py`
- Create: `tests/test_cli.py`

### Task 1: Escribir tests de integración

```python
# tests/test_cli.py
import os
import pytest
from typer.testing import CliRunner
from dqa.cli.main import app

runner = CliRunner()


def test_help_shows_compare_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "compare" in result.output


def test_compare_stable_exits_0(reference_parquet):
    """Comparar un dataset consigo mismo no debe disparar drift."""
    result = runner.invoke(app, ["compare", reference_parquet, reference_parquet])
    assert result.exit_code == 0


def test_compare_drifted_exits_1(reference_parquet, production_parquet):
    """price tiene drift de ~2.5σ — debe disparar alert y exit 1."""
    result = runner.invoke(app, [
        "compare", reference_parquet, production_parquet,
        "--fail-on", "alert",
    ])
    assert result.exit_code == 1


def test_fail_on_never_always_exits_0(reference_parquet, production_parquet):
    result = runner.invoke(app, [
        "compare", reference_parquet, production_parquet,
        "--fail-on", "never",
    ])
    assert result.exit_code == 0


def test_compare_unsupported_format_exits_2(tmp_path):
    path = str(tmp_path / "file.xlsx")
    open(path, "w").close()
    result = runner.invoke(app, ["compare", path, path])
    assert result.exit_code == 2


def test_compare_markdown_creates_file(reference_parquet, tmp_path):
    out = str(tmp_path / "report.md")
    result = runner.invoke(app, [
        "compare", reference_parquet, reference_parquet,
        "--format", "markdown", "--output", out,
    ])
    assert result.exit_code == 0
    assert os.path.exists(out)
    with open(out) as f:
        assert "DQA Drift Report" in f.read()


def test_compare_specific_columns(reference_parquet):
    result = runner.invoke(app, [
        "compare", reference_parquet, reference_parquet,
        "--columns", "age",
    ])
    assert result.exit_code == 0
```

### Task 2: Correr — debe fallar

```bash
uv run pytest tests/test_cli.py -v
```

### Task 3: Implementar cli/main.py

```python
# dqa/cli/main.py
"""Punto de entrada de la CLI. Cablea adapters y delega al engine."""
from __future__ import annotations
from enum import Enum
from typing import Optional
import typer
from dqa.engine import run_analysis
from dqa.adapters.readers.polars_reader import PolarsReader
from dqa.adapters.reporters.rich_reporter import RichReporter
from dqa.adapters.reporters.markdown_reporter import MarkdownReporter
from dqa.domain.models import DriftLevel

app = typer.Typer(
    name="dqa",
    help="Detecta drift estadístico entre dos datasets.",
    add_completion=False,
)


class OutputFormat(str, Enum):
    terminal = "terminal"
    markdown = "markdown"


class FailOn(str, Enum):
    alert   = "alert"
    warning = "warning"
    never   = "never"


_FAIL_THRESHOLDS: dict[FailOn, DriftLevel | None] = {
    FailOn.alert:   DriftLevel.ALERT,
    FailOn.warning: DriftLevel.WARNING,
    FailOn.never:   None,
}


@app.command()
def compare(
    reference:  str          = typer.Argument(..., help="Dataset de referencia (entrenamiento)."),
    production: str          = typer.Argument(..., help="Dataset de producción."),
    columns:    Optional[str] = typer.Option(None,       "--columns",  "-c", help="Columnas a analizar, separadas por coma."),
    format:     OutputFormat  = typer.Option("terminal", "--format",   "-f", help="Formato de salida."),
    output:     Optional[str] = typer.Option(None,       "--output",   "-o", help="Archivo de salida (con --format markdown)."),
    fail_on:    FailOn        = typer.Option("alert",    "--fail-on",        help="Nivel mínimo que dispara exit code 1."),
    bayesian:   bool          = typer.Option(False,      "--bayesian",       help="Activa análisis bayesiano (requiere uv sync --extra bayesian)."),
) -> None:
    """Compara dos datasets y reporta drift estadístico por columna."""
    col_list     = [c.strip() for c in columns.split(",")] if columns else None
    extra        = _load_bayesian_analyzer() if bayesian else []

    try:
        report = run_analysis(
            reference, production,
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
    """Importa BayesianAnalyzer en runtime para no penalizar el startup si no se usa."""
    try:
        from dqa.adapters.bayesian.pymc_analyzer import BayesianAnalyzer
        return [BayesianAnalyzer()]
    except ImportError:
        typer.echo(
            "⚠️  --bayesian requiere: uv sync --extra bayesian",
            err=True,
        )
        raise typer.Exit(code=2)
```

### Task 4: Correr — deben pasar

```bash
uv run pytest tests/test_cli.py -v
# Esperado: 7 passed
```

### Task 5: Verificar el comando instalado

```bash
uv run dqa --help
uv run dqa compare --help
```

### Task 6: Correr suite completa

```bash
uv run pytest -v
# Esperado: todos los tests pasan
```

### Task 7: Commit y push

```bash
touch dqa/cli/__init__.py
git add dqa/cli/ tests/test_cli.py
git commit -m "feat: CLI Typer — compare con exit codes para CI/CD"
git push origin feat/07-cli
```

**PR description:**
```
## Summary
- Comando `dqa compare ref prod` con Typer
- Exit codes: 0 = stable/warning, 1 = drift (configurable con --fail-on)
- --format terminal|markdown, --output para guardar archivo
- --columns para subconjunto de columnas
- --bayesian lazy-importa BayesianAnalyzer (no penaliza startup)
- Error descriptivo (exit 2) para formatos no soportados

## Test plan
- [ ] `uv run pytest tests/test_cli.py -v` → 7 passed
- [ ] `uv run pytest -v` (suite completa) → todos passed
- [ ] `uv run dqa --help` muestra subcomandos
- [ ] `uv run dqa compare --help` muestra todas las opciones
```

---

## Branch `feat/08-bayesian` — Adapter PyMC (opcional)

**PR title:** `feat: layer 3 — adapter bayesiano con HPD overlap (requiere dqa[bayesian])`

**Prerequisito:** `uv sync --extra bayesian`

**Files:**
- Create: `dqa/adapters/bayesian/__init__.py`
- Create: `dqa/adapters/bayesian/pymc_analyzer.py`
- Create: `tests/adapters/test_bayesian.py`

### Task 1: Escribir tests (marcados `slow`)

```python
# tests/adapters/test_bayesian.py
import numpy as np
import pytest

pytest.importorskip("pymc", reason="Requiere: uv sync --extra bayesian")

from dqa.adapters.bayesian.pymc_analyzer import BayesianAnalyzer
from dqa.domain.models import DriftLevel

RNG = np.random.default_rng(42)


@pytest.mark.slow
def test_stable_for_same_distribution():
    ref  = RNG.normal(0, 1, 300)
    prod = RNG.normal(0, 1, 300)
    result = BayesianAnalyzer().analyze(ref, prod)
    assert result.details["hdi_overlap"] > 0.5
    assert result.level in (DriftLevel.STABLE, DriftLevel.WARNING)


@pytest.mark.slow
def test_alert_for_large_drift():
    ref  = RNG.normal(0, 1, 300)
    prod = RNG.normal(5, 1, 300)
    result = BayesianAnalyzer().analyze(ref, prod)
    assert result.details["hdi_overlap"] < 0.2
    assert result.level == DriftLevel.ALERT


@pytest.mark.slow
def test_rhat_below_convergence_threshold():
    """r-hat < 1.05 indica que las cadenas MCMC convergieron correctamente."""
    ref  = RNG.normal(0, 1, 300)
    prod = RNG.normal(0, 1, 300)
    result = BayesianAnalyzer().analyze(ref, prod)
    assert result.details["rhat_mu_ref"]  < 1.05
    assert result.details["rhat_mu_prod"] < 1.05
```

### Task 2: Correr — debe fallar

```bash
uv sync --extra bayesian
uv run pytest tests/adapters/test_bayesian.py -v -m slow
```

### Task 3: Implementar pymc_analyzer.py

```python
# dqa/adapters/bayesian/pymc_analyzer.py
"""
Adapter bayesiano opcional: ajusta Normal(μ, σ) a cada dataset con PyMC
y compara los parámetros posteriores usando HPD overlap.

Solo se importa si el usuario usa --bayesian y tiene dqa[bayesian] instalado.
"""
from __future__ import annotations
import numpy as np
import pymc as pm
import arviz as az
from dqa.domain.models import AnalysisResult, DriftLevel


class BayesianAnalyzer:
    """
    Implementa ColumnAnalyzer usando inferencia bayesiana con PyMC.

    En lugar de métricas escalares, ajusta una distribución paramétrica a cada
    dataset y compara los intervalos de alta densidad posterior (HDI) de μ.
    Esto permite responder: "el μ de producción está fuera del rango plausible
    del μ de referencia", lo cual es más informativo que un p-value.
    """
    name = "bayesian"

    def analyze(self, ref: np.ndarray, prod: np.ndarray) -> AnalysisResult:
        """
        Ajusta Normal(μ, σ) a ref y prod por separado y compara sus posteriors de μ.

        La métrica principal es el overlap del HDI 94% entre μ_ref y μ_prod:
        - overlap > 0.6 → STABLE
        - overlap 0.3–0.6 → WARNING
        - overlap < 0.3 o μ_prod fuera del HDI de μ_ref → ALERT

        Args:
            ref:  Array de la distribución de referencia.
            prod: Array de la distribución de producción.

        Returns:
            AnalysisResult con mu_means, HDIs, hdi_overlap y r-hat de ambos fits.
        """
        idata_ref  = self._fit_normal(ref,  "ref")
        idata_prod = self._fit_normal(prod, "prod")

        mu_ref  = idata_ref.posterior["mu"].values.flatten()
        mu_prod = idata_prod.posterior["mu"].values.flatten()

        hdi_ref  = az.hdi(mu_ref,  hdi_prob=0.94)
        hdi_prod = az.hdi(mu_prod, hdi_prob=0.94)

        # Proporción del rango total que comparten ambos HDIs
        overlap = max(0.0, min(hdi_ref[1], hdi_prod[1]) - max(hdi_ref[0], hdi_prod[0]))
        span    = max(hdi_ref[1], hdi_prod[1]) - min(hdi_ref[0], hdi_prod[0]) + 1e-9
        overlap_ratio = overlap / span

        mu_prod_mean = float(mu_prod.mean())
        prod_outside_ref_hdi = mu_prod_mean < hdi_ref[0] or mu_prod_mean > hdi_ref[1]

        if overlap_ratio < 0.3 or prod_outside_ref_hdi:
            level = DriftLevel.ALERT
        elif overlap_ratio < 0.6:
            level = DriftLevel.WARNING
        else:
            level = DriftLevel.STABLE

        return AnalysisResult(
            level=level,
            details={
                "mu_ref_mean":   round(float(mu_ref.mean()), 4),
                "mu_prod_mean":  round(mu_prod_mean, 4),
                "hdi_ref":       [round(float(hdi_ref[0]), 4),  round(float(hdi_ref[1]), 4)],
                "hdi_prod":      [round(float(hdi_prod[0]), 4), round(float(hdi_prod[1]), 4)],
                "hdi_overlap":   round(overlap_ratio, 4),
                "rhat_mu_ref":   round(float(az.rhat(idata_ref)["mu"].values),  4),
                "rhat_mu_prod":  round(float(az.rhat(idata_prod)["mu"].values), 4),
            },
        )

    @staticmethod
    def _fit_normal(data: np.ndarray, tag: str) -> az.InferenceData:
        """Ajusta Normal(μ, σ) a data con 2 cadenas MCMC. Guarda en /tmp para auditoría."""
        with pm.Model():
            mu    = pm.Normal("mu",    mu=float(data.mean()), sigma=float(data.std() * 2))
            sigma = pm.HalfNormal("sigma", sigma=float(data.std()))
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
            idata = pm.sample(
                draws=500, tune=500, chains=2,
                nuts_sampler="nutpie",
                random_seed=42,
                progressbar=False,
            )
        idata.to_netcdf(f"/tmp/dqa_bayesian_{tag}.nc")
        return idata
```

### Task 4: Correr — deben pasar

```bash
uv run pytest tests/adapters/test_bayesian.py -v -m slow
# Esperado: 3 passed (puede tomar 30-90 segundos)
```

### Task 5: Verificar que CI normal no los corre

```bash
uv run pytest -v -m "not slow"
# Esperado: test_bayesian.py no aparece en la ejecución
```

### Task 6: Commit y push

```bash
touch dqa/adapters/bayesian/__init__.py
git add dqa/adapters/bayesian/ tests/adapters/test_bayesian.py
git commit -m "feat: layer 3 — adapter bayesiano PyMC con HPD overlap (opcional)"
git push origin feat/08-bayesian
```

**PR description:**
```
## Summary
- `BayesianAnalyzer` implementa ColumnAnalyzer con PyMC + nutpie
- Ajusta Normal(μ, σ) a ref y prod por separado, compara HDI 94% de μ
- Métricas: mu_means, HDIs, hdi_overlap, r-hat (convergencia)
- Guarda InferenceData en /tmp para auditoría posterior
- Solo se importa en runtime con --bayesian (no penaliza startup)
- Tests marcados con @pytest.mark.slow, excluidos del CI básico

## Test plan
- [ ] `uv sync --extra bayesian && uv run pytest -m slow -v` → 3 passed
- [ ] `uv run pytest -m "not slow"` (sin bayesian instalado) → todos passed
- [ ] r-hat < 1.05 para datos con n=300
```

---

## Branch `feat/09-ci` — GitHub Actions

**PR title:** `ci: GitHub Actions — tests + DQA self-check en cada PR`

**Files:**
- Create: `.github/workflows/ci.yml`

### Task 1: Crear el workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync   # instala core + dev, NO instala [bayesian]

      - name: Run tests (excluding slow)
        run: uv run pytest -m "not slow" --cov=dqa --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
        continue-on-error: true

  dqa-selfcheck:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - run: uv sync

      - name: Generate fixture data
        run: |
          uv run python - <<'EOF'
          import numpy as np
          import polars as pl
          rng = np.random.default_rng(42)
          pl.DataFrame({
              "age":   rng.normal(35, 10, 500).tolist(),
              "price": rng.normal(100, 20, 500).tolist(),
          }).write_parquet("ref.parquet")
          pl.DataFrame({
              "age":   rng.normal(35, 10, 500).tolist(),
              "price": rng.normal(100, 20, 500).tolist(),
          }).write_parquet("prod_stable.parquet")
          EOF

      - name: DQA self-check (exit 0 esperado — sin drift)
        run: uv run dqa compare ref.parquet prod_stable.parquet --format markdown --output dqa_report.md

      - name: Upload DQA report
        uses: actions/upload-artifact@v4
        with:
          name: dqa-report
          path: dqa_report.md
```

### Task 2: Commit y push

```bash
git add .github/workflows/ci.yml
git commit -m "ci: GitHub Actions — tests y DQA self-check"
git push origin feat/09-ci
```

**PR description:**
```
## Summary
- Job `test`: matrix Python 3.11/3.12, `uv sync` (sin bayesian), `pytest -m "not slow"`
- Job `dqa-selfcheck`: genera fixtures en runtime y corre `dqa compare` como smoke test
- Usa `astral-sh/setup-uv` para máxima velocidad de instalación
- Sube el reporte Markdown como artifact del run

## Test plan
- [ ] Verificar que el workflow pasa en la branch actual
- [ ] Confirmar que los jobs de matrix corren en paralelo
- [ ] Verificar que dqa_report.md aparece en los artifacts del run
```

---

## Criterios de éxito del MVP

| Criterio | Verificación |
|----------|-------------|
| `dqa compare ref.parquet prod.parquet` en < 5s para 100K filas | `time uv run dqa compare ref.parquet prod.parquet` |
| KS + PSI + JS detectan drift de 2σ en 100% con n=500 | Tests de `tests/analysis/` |
| Exit code 0 para datos estables, 1 para drift | Tests de `tests/test_cli.py` |
| `uv sync` instala sin PyMC | `uv sync && uv run pip list \| grep pymc` (no debe aparecer) |
| Suite completa pasa sin PyMC instalado | `uv run pytest -m "not slow"` en entorno limpio |
| Reporte Markdown válido generado | `uv run dqa compare ref prod --format markdown` |
| Schema drift detectado y reportado | Test `test_schema_diff_included_in_report` |

---

## Fuera del MVP (v0.2+)

- **Columnas categóricas**: TVD + chi-square — el adapter es el mismo, solo nuevos analyzers.
- **Reporte HTML**: agregar Jinja2 al extras group `[html]`. El esqueleto de MarkdownReporter ya anticipa el pattern.
- **Thresholds por columna**: nuevo comando `dqa config init` genera `dqa.yaml` con valores por defecto.
- **Profiling de un solo dataset**: comando `dqa profile dataset.parquet` — nuevo subcomando, sin tocar el engine.
- **Excel / Avro / Delta**: nuevos métodos en `PolarsReader` sin tocar el dominio.
