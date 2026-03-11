"""
Modelos de datos del dominio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class DriftLevel(IntEnum):
    """
    Nivel de severidad del drift detectado en una columna o dataset.
    """

    STABLE = 0
    WARNING = 1
    ALERT = 2

    def __str__(self) -> str:
        return self.name


@dataclass
class AnalysisResult:
    """
    Resultado de aplicar un único análisis estadístico a un par de columnas.
    """

    level: DriftLevel
    details: dict[str, Any]


@dataclass
class ColumnReport:
    """
    Reporte de drift para una columna individual, agregando todos sus análisis.
    """

    name: str
    dtype: str
    results: dict[
        str, AnalysisResult
    ]  # clave = nombre del análisis (ks_test, psi, etc.)

    @property
    def worst_level(self) -> DriftLevel:
        if not self.results:
            return DriftLevel.STABLE
        return max(r.level for r in self.results.values())


@dataclass
class SchemaDiff:
    """
    Diferencias estructurales entre dos datasets (columnas agregadas, removidas, tipo cambiado).
    """

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    type_changed: dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # col -> (dtype_ref, dtype_prod)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.type_changed)


@dataclass
class DatasetReport:
    """
    Reporte completo de drift a nivel de dataset.
    """

    columns: list[ColumnReport]
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
