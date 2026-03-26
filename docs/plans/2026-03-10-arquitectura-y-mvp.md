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

## Fuera del MVP (v0.2+)

- **Columnas categóricas**: TVD + chi-square — el adapter es el mismo, solo nuevos analyzers.
- **Reporte HTML**: agregar Jinja2 al extras group `[html]`. El esqueleto de MarkdownReporter ya anticipa el pattern.
- **Thresholds por columna**: nuevo comando `dqa config init` genera `dqa.yaml` con valores por defecto.
- **Profiling de un solo dataset**: comando `dqa profile dataset.parquet` — nuevo subcomando, sin tocar el engine.
- **Excel / Avro / Delta**: nuevos métodos en `PolarsReader` sin tocar el dominio.
