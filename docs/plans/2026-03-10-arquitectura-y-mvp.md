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

---

## Branch `feat/01-scaffolding` — Reparar configuración inicial

**PR title:** `fix: corregir pyproject.toml y pytest.ini para uv y estructura real del proyecto`

**Problema:** `pyproject.toml` tiene configuración de setuptools incorrecta (buscaría paquetes en `dqa/dqa/`) y `pytest.ini` apunta a `--cov=src` que no existe.

**Files:**
- Modify: `pyproject.toml`
- Modify: `pytest.ini`

### Task 1: Corregir pyproject.toml

Reemplazar el contenido completo:

```toml
[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "causal-data-quality-auditor"
version = "0.1.0"
description = "CLI que detecta drift estadístico entre datasets de producción y entrenamiento."
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [
    {name = "Gerardo Toboso", email = "gerardotoboso1909@gmail.com"}
]

# Dependencias core: rápidas de instalar, sin compilación pesada
dependencies = [
    "polars>=1.0.0",
    "scipy>=1.13.0",
    "numpy>=1.26.0",
    "rich>=13.7.0",
    "typer>=0.12.0",
    "pydantic>=2.7.0",
]

[project.optional-dependencies]
bayesian = [
    "pymc>=5.15.0",
    "nutpie>=0.13.0",
    "arviz>=0.19.0",
]
html = [
    "jinja2>=3.1.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-check>=2.6.2",
    "pytest-cov>=7.0.0",
    "pytest-html==3.2.0",
]

[project.scripts]
dqa = "dqa.cli.main:app"

[tool.setuptools.packages.find]
where = ["."]
include = ["dqa*"]
```

### Task 2: Corregir pytest.ini

```ini
[pytest]

addopts =
    -v
    --strict-markers
    --color=yes
    --cov=dqa
    --cov-report=term-missing
    --cov-report=term:skip-covered

testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning

markers =
    unit: Tests unitarios aislados
    integration: Tests que usan I/O real (archivos)
    slow: Tests lentos (PyMC sampling)

[coverage:run]
source = dqa
omit =
    */tests/*
    */__pycache__/*

[coverage:report]
precision = 2
show_missing = True
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if TYPE_CHECKING:
    @(abc\.)?abstractmethod
```

### Task 3: Verificar que el entorno arranca

```bash
uv sync
uv run pytest --collect-only
# Esperado: "no tests ran" sin errores de configuración
```

### Task 4: Commit y push

```bash
git add pyproject.toml pytest.ini
git commit -m "fix: corregir pyproject.toml y pytest.ini para uv y estructura real"
git push origin feat/01-scaffolding
```

**PR description:**
```
## Summary
- Corrige `package-dir` en setuptools (apuntaba a `dqa/dqa/`)
- Corrige `--cov=src` → `--cov=dqa` en pytest.ini
- Estructura `[project.optional-dependencies]` para extras `bayesian` e `html`
- `[dependency-groups]` para deps de desarrollo (compatible con uv)
- Entry point `dqa` registrado vía `[project.scripts]`

## Test plan
- [ ] `uv sync` instala sin errores
- [ ] `uv run pytest --collect-only` corre sin errores de configuración
- [ ] `uv run dqa --help` falla gracefully (no existe aún) sin ImportError de setuptools
```

---

## Branch `feat/02-domain` — Modelos y ports del dominio

**PR title:** `feat: dominio puro — modelos de datos y protocols sin dependencias externas`

**Files:**
- Create: `dqa/__init__.py`
- Create: `dqa/domain/__init__.py`
- Create: `dqa/domain/models.py`
- Create: `dqa/domain/ports.py`
- Create: `tests/__init__.py`
- Create: `tests/domain/__init__.py`
- Create: `tests/domain/test_models.py`

### Task 1: Escribir los tests primero

```python
# tests/domain/test_models.py
import pytest
from dqa.domain.models import DriftLevel, ColumnReport, DatasetReport, AnalysisResult, SchemaDiff


def test_drift_level_ordering():
    assert DriftLevel.STABLE < DriftLevel.WARNING < DriftLevel.ALERT


def test_column_report_worst_level_no_results():
    report = ColumnReport(name="price", dtype="float64", results={})
    assert report.worst_level == DriftLevel.STABLE


def test_column_report_worst_level_with_alert():
    report = ColumnReport(
        name="price",
        dtype="float64",
        results={
            "ks_test": AnalysisResult(level=DriftLevel.STABLE,  details={}),
            "psi":     AnalysisResult(level=DriftLevel.ALERT,   details={}),
        },
    )
    assert report.worst_level == DriftLevel.ALERT


def test_dataset_report_has_alerts():
    col_ok    = ColumnReport(name="age",   dtype="float64",
                             results={"psi": AnalysisResult(level=DriftLevel.STABLE, details={})})
    col_alert = ColumnReport(name="price", dtype="float64",
                             results={"psi": AnalysisResult(level=DriftLevel.ALERT,  details={})})
    report = DatasetReport(columns=[col_ok, col_alert])
    assert report.has_alerts is True
    assert report.alert_columns == ["price"]


def test_dataset_report_overall_level():
    col = ColumnReport(name="x", dtype="float64",
                       results={"psi": AnalysisResult(level=DriftLevel.WARNING, details={})})
    report = DatasetReport(columns=[col])
    assert report.overall_level == DriftLevel.WARNING


def test_schema_diff_has_changes():
    diff = SchemaDiff(added=["new_col"])
    assert diff.has_changes is True


def test_schema_diff_no_changes():
    diff = SchemaDiff()
    assert diff.has_changes is False
```

### Task 2: Correr — debe fallar

```bash
uv run pytest tests/domain/test_models.py -v
# Esperado: ImportError (módulo no existe aún)
```

### Task 3: Implementar models.py

```python
# dqa/domain/models.py
"""Modelos de datos del dominio. Sin dependencias externas."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class DriftLevel(IntEnum):
    """Nivel de severidad del drift detectado en una columna o dataset."""
    STABLE  = 0
    WARNING = 1
    ALERT   = 2

    def __str__(self) -> str:
        return self.name


@dataclass
class AnalysisResult:
    """Resultado de aplicar un único análisis estadístico a un par de columnas."""
    level:   DriftLevel
    details: dict[str, Any]


@dataclass
class ColumnReport:
    """Reporte de drift para una columna individual, agregando todos sus análisis."""
    name:    str
    dtype:   str
    results: dict[str, AnalysisResult]   # clave = nombre del análisis (ks_test, psi, etc.)

    @property
    def worst_level(self) -> DriftLevel:
        if not self.results:
            return DriftLevel.STABLE
        return max(r.level for r in self.results.values())


@dataclass
class SchemaDiff:
    """Diferencias estructurales entre dos datasets (columnas, tipos)."""
    added:        list[str]                  = field(default_factory=list)
    removed:      list[str]                  = field(default_factory=list)
    type_changed: dict[str, tuple[str, str]] = field(default_factory=dict)  # col -> (dtype_ref, dtype_prod)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.type_changed)


@dataclass
class DatasetReport:
    """Reporte completo de drift a nivel de dataset."""
    columns:     list[ColumnReport]
    schema_diff: SchemaDiff = field(default_factory=SchemaDiff)

    @property
    def has_alerts(self) -> bool:
        return any(c.worst_level == DriftLevel.ALERT for c in self.columns)

    @property
    def alert_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.worst_level == DriftLevel.ALERT]

    @property
    def overall_level(self) -> DriftLevel:
        if not self.columns:
            return DriftLevel.STABLE
        return max(c.worst_level for c in self.columns)
```

### Task 4: Implementar ports.py

```python
# dqa/domain/ports.py
"""Interfaces (Protocols) que desacoplan el engine de sus implementaciones concretas."""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from dqa.domain.models import AnalysisResult, DatasetReport


@runtime_checkable
class DataReader(Protocol):
    """Lee un dataset y lo expone como columnas numéricas en arrays numpy."""

    def read(self, path: str) -> dict[str, np.ndarray]:
        """Retorna solo las columnas numéricas del dataset como {col: array}."""
        ...

    def schema(self, path: str) -> dict[str, str]:
        """Retorna el schema completo del dataset como {col: dtype_str}."""
        ...


@runtime_checkable
class ColumnAnalyzer(Protocol):
    """Aplica un análisis estadístico a un par de arrays (ref, prod)."""
    name: str

    def analyze(self, ref: np.ndarray, prod: np.ndarray) -> AnalysisResult:
        """Compara ref contra prod y retorna el nivel de drift detectado."""
        ...


@runtime_checkable
class Reporter(Protocol):
    """Emite el reporte final de drift en el formato que corresponda."""

    def report(self, result: DatasetReport) -> None:
        ...
```

### Task 5: Crear `__init__.py` vacíos

```bash
touch dqa/__init__.py dqa/domain/__init__.py
mkdir -p tests/domain && touch tests/__init__.py tests/domain/__init__.py
```

### Task 6: Correr — deben pasar

```bash
uv run pytest tests/domain/test_models.py -v
# Esperado: 6 passed
```

### Task 7: Commit y push

```bash
git add dqa/ tests/
git commit -m "feat: dominio puro — DriftLevel, ColumnReport, DatasetReport, Protocols"
git push origin feat/02-domain
```

**PR description:**
```
## Summary
- Modelos inmutables del dominio: `DriftLevel`, `AnalysisResult`, `ColumnReport`,
  `SchemaDiff`, `DatasetReport`
- Protocols `DataReader`, `ColumnAnalyzer`, `Reporter` — definen contratos sin
  acoplar a ninguna librería concreta
- Cero dependencias externas en `dqa/domain/`

## Test plan
- [ ] `uv run pytest tests/domain/ -v` → 6 passed
- [ ] Confirmar que `dqa/domain/` no importa polars, scipy, pymc ni rich
```

---

## Branch `feat/03-analysis-core` — Capas de análisis estadístico

**PR title:** `feat: análisis estadístico — schema drift, KS+PSI y KL/JS Divergence`

Agrupa los tres módulos de análisis puro (sin adapters) en una sola branch porque forman una unidad cohesiva y ninguno depende de los otros.

**Files:**
- Create: `dqa/analysis/__init__.py`
- Create: `dqa/analysis/schema.py`
- Create: `dqa/analysis/classical.py`
- Create: `dqa/analysis/information.py`
- Create: `tests/analysis/__init__.py`
- Create: `tests/analysis/test_schema.py`
- Create: `tests/analysis/test_classical.py`
- Create: `tests/analysis/test_information.py`

---

### Task 1: Tests de schema drift

```python
# tests/analysis/test_schema.py
from dqa.analysis.schema import detect_schema_drift, common_numeric_columns
from dqa.domain.models import SchemaDiff


def test_identical_schemas_no_diff():
    ref  = {"age": "float64", "price": "float64"}
    prod = {"age": "float64", "price": "float64"}
    assert not detect_schema_drift(ref, prod).has_changes


def test_detects_added_column():
    diff = detect_schema_drift({"age": "float64"}, {"age": "float64", "score": "float64"})
    assert "score" in diff.added
    assert not diff.removed


def test_detects_removed_column():
    diff = detect_schema_drift({"age": "float64", "price": "float64"}, {"age": "float64"})
    assert "price" in diff.removed
    assert not diff.added


def test_detects_type_change():
    diff = detect_schema_drift({"age": "float64"}, {"age": "int32"})
    assert diff.type_changed["age"] == ("float64", "int32")


def test_common_numeric_columns_intersection():
    ref  = {"age": "float64", "name": "str", "price": "float64"}
    prod = {"age": "float64", "name": "str", "score": "float64"}
    assert common_numeric_columns(ref, prod) == ["age"]


def test_common_numeric_columns_excludes_type_mismatch():
    # "age" existe en ambos pero con dtype distinto → no se incluye
    ref  = {"age": "float64"}
    prod = {"age": "int32"}
    assert common_numeric_columns(ref, prod) == []
```

### Task 2: Correr schema tests — debe fallar

```bash
uv run pytest tests/analysis/test_schema.py -v
```

### Task 3: Implementar schema.py

```python
# dqa/analysis/schema.py
"""Layer 0: detección de schema drift. Sin dependencias externas."""
from __future__ import annotations
from dqa.domain.models import SchemaDiff

_NUMERIC_DTYPES = frozenset({
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
})


def detect_schema_drift(ref: dict[str, str], prod: dict[str, str]) -> SchemaDiff:
    """
    Compara dos schemas (col → dtype_str) y retorna sus diferencias estructurales.

    Args:
        ref:  Schema del dataset de referencia {columna: dtype}.
        prod: Schema del dataset de producción {columna: dtype}.

    Returns:
        SchemaDiff con columnas agregadas, removidas y con tipo cambiado.
    """
    ref_cols  = set(ref)
    prod_cols = set(prod)

    type_changed = {
        col: (ref[col], prod[col])
        for col in ref_cols & prod_cols
        if ref[col] != prod[col]
    }

    return SchemaDiff(
        added=sorted(prod_cols - ref_cols),
        removed=sorted(ref_cols - prod_cols),
        type_changed=type_changed,
    )


def common_numeric_columns(ref: dict[str, str], prod: dict[str, str]) -> list[str]:
    """
    Retorna las columnas numéricas presentes en ambos datasets con el mismo dtype.

    Solo estas columnas son elegibles para análisis estadístico de distribuciones.
    """
    return sorted(
        col
        for col in set(ref) & set(prod)
        if ref[col] in _NUMERIC_DTYPES
        and ref[col] == prod[col]
    )
```

### Task 4: Correr — deben pasar

```bash
uv run pytest tests/analysis/test_schema.py -v
# Esperado: 6 passed
```

### Task 5: Commit parcial

```bash
git add dqa/analysis/schema.py tests/analysis/test_schema.py dqa/analysis/__init__.py tests/analysis/__init__.py
git commit -m "feat: layer 0 — schema drift sin dependencias externas"
```

---

### Task 6: Tests de KS test y PSI

```python
# tests/analysis/test_classical.py
import numpy as np
import pytest
from dqa.analysis.classical import ks_test, population_stability_index
from dqa.domain.models import DriftLevel

RNG = np.random.default_rng(42)
REF        = RNG.normal(loc=0,   scale=1, size=2000)
PROD_SAME  = RNG.normal(loc=0,   scale=1, size=2000)
PROD_DRIFT = RNG.normal(loc=3.0, scale=1, size=2000)


def test_ks_stable_for_same_distribution():
    result = ks_test(REF, PROD_SAME)
    assert result.level == DriftLevel.STABLE
    assert result.details["p_value"] > 0.05


def test_ks_alert_for_drifted_distribution():
    result = ks_test(REF, PROD_DRIFT)
    assert result.level == DriftLevel.ALERT
    assert result.details["p_value"] < 0.05


def test_ks_result_has_required_keys():
    result = ks_test(REF, PROD_SAME)
    assert "statistic" in result.details
    assert "p_value"   in result.details


def test_psi_stable_for_identical_data():
    result = population_stability_index(REF, REF)
    assert result.level == DriftLevel.STABLE
    assert result.details["psi"] < 0.1


def test_psi_alert_for_large_drift():
    result = population_stability_index(REF, PROD_DRIFT)
    assert result.level == DriftLevel.ALERT
    assert result.details["psi"] > 0.2


def test_psi_warning_band():
    prod_moderate = RNG.normal(loc=0.8, scale=1, size=2000)
    result = population_stability_index(REF, prod_moderate)
    assert result.level >= DriftLevel.WARNING
```

### Task 7: Correr — debe fallar

```bash
uv run pytest tests/analysis/test_classical.py -v
```

### Task 8: Implementar classical.py

```python
# dqa/analysis/classical.py
"""Layer 1: tests clásicos de drift — KS test y PSI. Requiere scipy."""
from __future__ import annotations
import numpy as np
from scipy import stats
from dqa.domain.models import AnalysisResult, DriftLevel


def ks_test(ref: np.ndarray, prod: np.ndarray) -> AnalysisResult:
    """
    Kolmogorov-Smirnov de dos muestras: ¿provienen de la misma distribución?

    Detecta diferencias en cualquier parte de la distribución (cola, centro, forma).
    Sensible a n pequeño; complementar con PSI para confirmación.

    Args:
        ref:  Array de la distribución de referencia.
        prod: Array de la distribución de producción.

    Returns:
        ALERT si p_value < 0.05, STABLE en caso contrario.
    """
    stat, p_value = stats.ks_2samp(ref, prod)
    level = DriftLevel.ALERT if p_value < 0.05 else DriftLevel.STABLE
    return AnalysisResult(
        level=level,
        details={"statistic": round(float(stat), 4), "p_value": round(float(p_value), 4)},
    )


def population_stability_index(
    ref: np.ndarray, prod: np.ndarray, n_bins: int = 10
) -> AnalysisResult:
    """
    Population Stability Index — estándar de la industria financiera para monitoreo.

    Los bins se calculan sobre los percentiles de `ref` para que la referencia
    defina la escala. Valores de producción fuera del rango de referencia caen
    en los bins extremos (–∞, +∞).

    Umbrales convencionales:
        PSI < 0.1  → STABLE  (sin cambio significativo)
        PSI 0.1–0.2 → WARNING (cambio moderado, monitorear)
        PSI > 0.2  → ALERT   (cambio significativo, investigar)

    Args:
        ref:    Array de referencia.
        prod:   Array de producción.
        n_bins: Número de bins (default 10, estándar de industria).

    Returns:
        AnalysisResult con el valor PSI y su nivel de severidad.
    """
    breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    ref_freq  = np.histogram(ref,  bins=breakpoints)[0] / len(ref)
    prod_freq = np.histogram(prod, bins=breakpoints)[0] / len(prod)

    # Reemplazar ceros para evitar log(0)
    ref_freq  = np.where(ref_freq  == 0, 1e-6, ref_freq)
    prod_freq = np.where(prod_freq == 0, 1e-6, prod_freq)

    psi = float(np.sum((prod_freq - ref_freq) * np.log(prod_freq / ref_freq)))

    if   psi < 0.1:  level = DriftLevel.STABLE
    elif psi < 0.2:  level = DriftLevel.WARNING
    else:            level = DriftLevel.ALERT

    return AnalysisResult(level=level, details={"psi": round(psi, 4)})
```

### Task 9: Correr — deben pasar

```bash
uv run pytest tests/analysis/test_classical.py -v
# Esperado: 6 passed
```

### Task 10: Commit parcial

```bash
git add dqa/analysis/classical.py tests/analysis/test_classical.py
git commit -m "feat: layer 1 — KS test y PSI"
```

---

### Task 11: Tests de KL/JS Divergence

```python
# tests/analysis/test_information.py
import time
import numpy as np
import pytest
from dqa.analysis.information import kl_js_divergence
from dqa.domain.models import DriftLevel

RNG = np.random.default_rng(42)


def test_js_near_zero_for_identical_data():
    data = RNG.normal(0, 1, 3000)
    result = kl_js_divergence(data, data)
    assert result.details["js_divergence"] < 0.01
    assert result.level == DriftLevel.STABLE


def test_js_high_for_disjoint_distributions():
    ref  = RNG.normal(loc=0,  scale=1, size=3000)
    prod = RNG.normal(loc=10, scale=1, size=3000)
    result = kl_js_divergence(ref, prod)
    assert result.details["js_divergence"] > 0.9
    assert result.level == DriftLevel.ALERT


def test_result_has_all_keys():
    data = RNG.normal(0, 1, 1000)
    result = kl_js_divergence(data, data)
    assert "kl_ref_to_prod" in result.details
    assert "kl_prod_to_ref" in result.details
    assert "js_divergence"  in result.details


def test_runs_under_500ms_for_large_arrays():
    ref  = RNG.normal(0, 1, 10_000)
    prod = RNG.normal(0, 1, 10_000)
    t0 = time.perf_counter()
    kl_js_divergence(ref, prod)
    assert time.perf_counter() - t0 < 0.5
```

### Task 12: Correr — debe fallar

```bash
uv run pytest tests/analysis/test_information.py -v
```

### Task 13: Implementar information.py

```python
# dqa/analysis/information.py
"""Layer 2: KL Divergence y Jensen-Shannon Divergence via KDE. Requiere scipy/numpy."""
from __future__ import annotations
import numpy as np
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
from dqa.domain.models import AnalysisResult, DriftLevel

_JS_WARNING_THRESHOLD = 0.05
_JS_ALERT_THRESHOLD   = 0.10


def kl_js_divergence(
    ref: np.ndarray, prod: np.ndarray, n_points: int = 500
) -> AnalysisResult:
    """
    Calcula KL Divergence y Jensen-Shannon Divergence entre dos distribuciones continuas.

    Usa Kernel Density Estimation (KDE gaussiano) para estimar las densidades de
    probabilidad antes de calcular las divergencias. JS es la métrica principal:
    es simétrica y acotada en [0, 1], a diferencia de KL que es asimétrica y
    puede ser infinita si los soportes no se superponen.

    JS = 0: distribuciones idénticas.
    JS = 1: distribuciones con soporte completamente disjunto.
    Umbral de alerta: JS > 0.10 (empírico, configurable por columna en v0.2).

    Args:
        ref:      Array de referencia (distribución base).
        prod:     Array de producción (distribución a comparar).
        n_points: Resolución de la grilla para KDE (trade-off velocidad/precisión).

    Returns:
        AnalysisResult con kl_ref_to_prod, kl_prod_to_ref, js_divergence y nivel.
    """
    x_grid = np.linspace(
        min(ref.min(), prod.min()),
        max(ref.max(), prod.max()),
        n_points,
    )

    p = gaussian_kde(ref)(x_grid)  + 1e-10
    q = gaussian_kde(prod)(x_grid) + 1e-10
    p, q = p / p.sum(), q / q.sum()

    m = (p + q) / 2
    kl_pq = float(np.sum(rel_entr(p, q)))
    kl_qp = float(np.sum(rel_entr(q, p)))
    js    = float(np.sum(rel_entr(p, m)) / 2 + np.sum(rel_entr(q, m)) / 2)

    if   js < _JS_WARNING_THRESHOLD: level = DriftLevel.STABLE
    elif js < _JS_ALERT_THRESHOLD:   level = DriftLevel.WARNING
    else:                             level = DriftLevel.ALERT

    return AnalysisResult(
        level=level,
        details={
            "kl_ref_to_prod": round(kl_pq, 4),
            "kl_prod_to_ref": round(kl_qp, 4),
            "js_divergence":  round(js, 4),
        },
    )
```

### Task 14: Correr — deben pasar

```bash
uv run pytest tests/analysis/ -v
# Esperado: todos los tests de analysis pasan
```

### Task 15: Commit y push

```bash
git add dqa/analysis/information.py tests/analysis/test_information.py
git commit -m "feat: layer 2 — KL/JS Divergence via KDE"
git push origin feat/03-analysis-core
```

**PR description:**
```
## Summary
- Layer 0 (`schema.py`): detecta columnas agregadas/removidas y tipos cambiados,
  sin dependencias externas
- Layer 1 (`classical.py`): KS test (scipy) y PSI con bins percentílicos sobre ref
- Layer 2 (`information.py`): KL/JS Divergence via KDE gaussiano (scipy)
- Docstrings en las tres funciones de análisis no triviales

## Test plan
- [ ] `uv run pytest tests/analysis/ -v` → todos passed
- [ ] Confirmar que `dqa/analysis/` no importa polars, rich ni pymc
- [ ] `test_runs_under_500ms_for_large_arrays` pasa en CI
```

---

## Branch `feat/04-polars-adapter` — Adapter de lectura de datos

**PR title:** `feat: adapter Polars — lectura de Parquet, CSV y NDJSON`

**Files:**
- Create: `dqa/adapters/__init__.py`
- Create: `dqa/adapters/readers/__init__.py`
- Create: `dqa/adapters/readers/polars_reader.py`
- Create: `tests/adapters/__init__.py`
- Create: `tests/adapters/test_polars_reader.py`
- Create: `tests/conftest.py`

### Task 1: Fixtures compartidas

```python
# tests/conftest.py
import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="session")
def reference_parquet(tmp_path_factory):
    rng = np.random.default_rng(42)
    df = pl.DataFrame({
        "age":   rng.normal(35, 10, 1000).tolist(),
        "price": rng.normal(100, 20, 1000).tolist(),
        "name":  ["Alice"] * 500 + ["Bob"] * 500,   # str — debe ignorarse
    })
    path = str(tmp_path_factory.mktemp("fixtures") / "reference.parquet")
    df.write_parquet(path)
    return path


@pytest.fixture(scope="session")
def production_parquet(tmp_path_factory):
    rng = np.random.default_rng(99)
    df = pl.DataFrame({
        "age":   rng.normal(35, 10, 1000).tolist(),    # sin drift
        "price": rng.normal(150, 20, 1000).tolist(),   # drift fuerte
        "name":  ["Alice"] * 1000,
    })
    path = str(tmp_path_factory.mktemp("fixtures") / "production.parquet")
    df.write_parquet(path)
    return path
```

### Task 2: Escribir tests del reader

```python
# tests/adapters/test_polars_reader.py
import numpy as np
import polars as pl
import pytest
from dqa.adapters.readers.polars_reader import PolarsReader
from dqa.domain.ports import DataReader


def test_implements_data_reader_protocol():
    assert isinstance(PolarsReader(), DataReader)


def test_read_parquet_returns_only_numeric_columns(reference_parquet):
    data = PolarsReader().read(reference_parquet)
    assert "age"   in data
    assert "price" in data
    assert "name"  not in data   # str — excluida


def test_read_returns_numpy_arrays(reference_parquet):
    data = PolarsReader().read(reference_parquet)
    assert all(isinstance(arr, np.ndarray) for arr in data.values())


def test_schema_returns_dtype_strings(reference_parquet):
    schema = PolarsReader().schema(reference_parquet)
    assert schema["age"]   in ("float64", "float32")
    assert schema["price"] in ("float64", "float32")
    assert "name" in schema   # schema incluye todas las columnas, no solo numéricas


def test_read_csv(tmp_path):
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    path = str(tmp_path / "test.csv")
    df.write_csv(path)
    data = PolarsReader().read(path)
    assert "x" in data and "y" in data


def test_read_ndjson(tmp_path):
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    path = str(tmp_path / "test.ndjson")
    df.write_ndjson(path)
    data = PolarsReader().read(path)
    assert "a" in data and "b" in data


def test_unsupported_format_raises(tmp_path):
    path = str(tmp_path / "test.xlsx")
    open(path, "w").close()
    with pytest.raises(ValueError, match="formato no soportado"):
        PolarsReader().read(path)
```

### Task 3: Correr — debe fallar

```bash
uv run pytest tests/adapters/test_polars_reader.py -v
```

### Task 4: Implementar polars_reader.py

```python
# dqa/adapters/readers/polars_reader.py
"""Adapter DataReader implementado con Polars. Soporta Parquet, CSV y NDJSON."""
from __future__ import annotations
import numpy as np
import polars as pl
from pathlib import Path
from dqa.domain.ports import DataReader

_NUMERIC_POLARS_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}


class PolarsReader:
    """Implementa DataReader usando Polars para máxima performance en formatos columnar."""

    def _load(self, path: str) -> pl.DataFrame:
        suffix = Path(path).suffix.lower()
        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix == ".csv":
            return pl.read_csv(path)
        if suffix in (".ndjson", ".jsonl"):
            return pl.read_ndjson(path)
        raise ValueError(
            f"formato no soportado: '{suffix}'. Formatos válidos: .parquet, .csv, .ndjson"
        )

    def read(self, path: str) -> dict[str, np.ndarray]:
        """Retorna solo las columnas numéricas del dataset como arrays numpy."""
        df = self._load(path)
        return {
            col: df[col].to_numpy()
            for col in df.columns
            if df[col].dtype in _NUMERIC_POLARS_DTYPES
        }

    def schema(self, path: str) -> dict[str, str]:
        """Retorna el schema completo (todas las columnas) como {col: dtype_str}."""
        df = self._load(path)
        return {col: str(df[col].dtype).lower() for col in df.columns}
```

### Task 5: Crear `__init__.py` faltantes

```bash
touch dqa/adapters/__init__.py dqa/adapters/readers/__init__.py
mkdir -p tests/adapters && touch tests/adapters/__init__.py
```

### Task 6: Correr — deben pasar

```bash
uv run pytest tests/adapters/test_polars_reader.py -v
# Esperado: 7 passed
```

### Task 7: Commit y push

```bash
git add dqa/adapters/ tests/adapters/test_polars_reader.py tests/conftest.py
git commit -m "feat: adapter Polars — lector de Parquet, CSV y NDJSON"
git push origin feat/04-polars-adapter
```

**PR description:**
```
## Summary
- `PolarsReader` implementa el Protocol `DataReader` del dominio
- Soporta `.parquet`, `.csv` y `.ndjson`
- `read()` filtra y retorna solo columnas numéricas como numpy arrays
- `schema()` retorna el schema completo (para schema drift)
- Error descriptivo para formatos no soportados

## Test plan
- [ ] `uv run pytest tests/adapters/test_polars_reader.py -v` → 7 passed
- [ ] Verificar que `isinstance(PolarsReader(), DataReader)` pasa (Protocol check)
- [ ] Fixtures de session en conftest.py disponibles para branches posteriores
```

---

## Branch `feat/05-engine` — Orquestador de análisis

**PR title:** `feat: engine — orquestador con inyección de dependencias`

**Files:**
- Create: `dqa/engine.py`
- Create: `tests/test_engine.py`

### Task 1: Escribir tests con mocks mínimos

```python
# tests/test_engine.py
import numpy as np
import pytest
from dqa.engine import run_analysis
from dqa.domain.models import DatasetReport, DriftLevel

RNG = np.random.default_rng(42)


class FakeReader:
    """Mock de DataReader que sirve datos en memoria sin I/O."""
    def __init__(self, ref_data: dict, prod_data: dict, schema: dict | None = None):
        self._data   = {"ref": ref_data, "prod": prod_data}
        self._schema = schema or {k: "float64" for k in ref_data}

    def read(self, path: str) -> dict:
        return self._data[path]

    def schema(self, path: str) -> dict:
        return self._schema


def make_reader(ref: dict, prod: dict, schema: dict | None = None) -> FakeReader:
    return FakeReader(ref, prod, schema)


def test_returns_dataset_report():
    ref  = {"price": RNG.normal(0, 1, 200)}
    prod = {"price": RNG.normal(0, 1, 200)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod))
    assert isinstance(report, DatasetReport)


def test_analyzes_all_common_numeric_columns():
    ref  = {"price": RNG.normal(0, 1, 200), "age": RNG.normal(30, 5, 200)}
    prod = {"price": RNG.normal(0, 1, 200), "age": RNG.normal(30, 5, 200)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod))
    names = {c.name for c in report.columns}
    assert "price" in names and "age" in names


def test_filters_to_requested_columns():
    ref  = {"price": RNG.normal(0, 1, 200), "age": RNG.normal(30, 5, 200)}
    prod = {"price": RNG.normal(0, 1, 200), "age": RNG.normal(30, 5, 200)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod), columns=["price"])
    assert len(report.columns) == 1
    assert report.columns[0].name == "price"


def test_detects_drift_on_drifted_column():
    ref  = {"price": RNG.normal(0, 1, 500)}
    prod = {"price": RNG.normal(10, 1, 500)}   # drift fuerte
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod))
    price = next(c for c in report.columns if c.name == "price")
    assert price.worst_level == DriftLevel.ALERT


def test_schema_diff_included_in_report():
    ref_schema  = {"price": "float64", "age": "float64"}
    prod_schema = {"price": "float64"}   # "age" fue removida

    class SchemaMismatchReader(FakeReader):
        def schema(self, path: str) -> dict:
            return ref_schema if path == "ref" else prod_schema

    reader = SchemaMismatchReader(
        {"price": RNG.normal(0, 1, 100)},
        {"price": RNG.normal(0, 1, 100)},
    )
    report = run_analysis("ref", "prod", reader=reader)
    assert "age" in report.schema_diff.removed


def test_each_column_has_all_default_analysis_keys():
    ref  = {"x": RNG.normal(0, 1, 200)}
    prod = {"x": RNG.normal(0, 1, 200)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod))
    assert len(report.columns) == 1
    keys = set(report.columns[0].results.keys())
    assert {"ks_test", "psi", "kl_js"} == keys
```

### Task 2: Correr — debe fallar

```bash
uv run pytest tests/test_engine.py -v
```

### Task 3: Implementar engine.py

```python
# dqa/engine.py
"""
Orquestador del análisis de drift.

Coordina la lectura de datos, la detección de schema drift y la aplicación
de los analizadores estadísticos por columna. No importa ningún adaptador
concreto — recibe todo por inyección de dependencias.
"""
from __future__ import annotations
from dqa.domain.models import ColumnReport, DatasetReport
from dqa.domain.ports import DataReader, ColumnAnalyzer
from dqa.analysis.schema import detect_schema_drift, common_numeric_columns
from dqa.analysis.classical import ks_test, population_stability_index
from dqa.analysis.information import kl_js_divergence


def _make_analyzer(analysis_name: str, fn):
    """Construye un objeto que cumple el Protocol ColumnAnalyzer desde una función."""
    return type("_Analyzer", (), {"name": analysis_name, "analyze": staticmethod(fn)})()


_DEFAULT_ANALYZERS: list[ColumnAnalyzer] = [
    _make_analyzer("ks_test", ks_test),
    _make_analyzer("psi",     population_stability_index),
    _make_analyzer("kl_js",   kl_js_divergence),
]


def run_analysis(
    ref_path:        str,
    prod_path:       str,
    *,
    reader:          DataReader,
    analyzers:       list[ColumnAnalyzer] | None = None,
    extra_analyzers: list[ColumnAnalyzer] | None = None,
    columns:         list[str] | None = None,
) -> DatasetReport:
    """
    Ejecuta el análisis completo de drift entre dos datasets.

    Primero corre schema drift (Layer 0, sin costo estadístico) y luego aplica
    cada analizador a las columnas numéricas comunes.

    Args:
        ref_path:        Ruta al dataset de referencia.
        prod_path:       Ruta al dataset de producción.
        reader:          Adapter que implementa DataReader.
        analyzers:       Analizadores a usar. Default: KS test, PSI, KL/JS.
        extra_analyzers: Analizadores adicionales (e.g. BayesianAnalyzer) que se
                         concatenan a los default sin reemplazarlos.
        columns:         Si se provee, analiza solo este subconjunto de columnas.

    Returns:
        DatasetReport con schema diff y un ColumnReport por columna analizada.
    """
    active_analyzers = (analyzers if analyzers is not None else _DEFAULT_ANALYZERS) + (extra_analyzers or [])

    ref_schema  = reader.schema(ref_path)
    prod_schema = reader.schema(prod_path)
    schema_diff = detect_schema_drift(ref_schema, prod_schema)

    target_cols = common_numeric_columns(ref_schema, prod_schema)
    if columns is not None:
        target_cols = [c for c in target_cols if c in columns]

    ref_data  = reader.read(ref_path)
    prod_data = reader.read(prod_path)

    column_reports = [
        ColumnReport(
            name=col,
            dtype=ref_schema[col],
            results={
                analyzer.name: analyzer.analyze(ref_data[col], prod_data[col])
                for analyzer in active_analyzers
            },
        )
        for col in target_cols
    ]

    return DatasetReport(columns=column_reports, schema_diff=schema_diff)
```

### Task 4: Correr — deben pasar

```bash
uv run pytest tests/test_engine.py -v
# Esperado: 6 passed
```

### Task 5: Correr toda la suite hasta aquí

```bash
uv run pytest -v
# Esperado: todos los tests previos también pasan
```

### Task 6: Commit y push

```bash
git add dqa/engine.py tests/test_engine.py
git commit -m "feat: engine — orquestador con inyección de dependencias"
git push origin feat/05-engine
```

**PR description:**
```
## Summary
- `run_analysis()` coordina schema diff + análisis estadístico por columna
- Sin imports de librerías concretas: todo entra por DI (`reader`, `analyzers`)
- Soporta `extra_analyzers` para agregar la capa bayesiana sin modificar el core
- `columns` param para analizar subconjunto específico

## Test plan
- [ ] `uv run pytest tests/test_engine.py -v` → 6 passed
- [ ] `uv run pytest -v` (suite completa) → todos passed
- [ ] Confirmar que `dqa/engine.py` no importa polars, rich ni pymc
```

---

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
