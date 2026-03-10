"""Interfaces (Protocols) que desacoplan el engine de sus implementaciones concretas."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from dqa.domain.models import AnalysisResult, DatasetReport


@runtime_checkable
class DataReader(Protocol):
    """Lee un dataset y expone sus columnas numéricas como arrays numpy."""

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
