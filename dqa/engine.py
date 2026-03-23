"""
Orquestador del análisis de drift.

Coordina la lectura de datos, la detección de schema drift y la aplicación
de los analizadores estadísticos por columna. No importa ningún adaptador
concreto — recibe todo por inyección de dependencias.
"""

from __future__ import annotations

from typing import Optional

from dqa.analysis.classical import ks_test, population_stability_index
from dqa.analysis.information import kl_js_divergence
from dqa.analysis.schema import common_numeric_columns, detect_schema_drift
from dqa.domain.models import ColumnReport, DatasetReport
from dqa.domain.ports import ColumnAnalyzer, DataReader


def _make_analyzer(analysis_name: str, fn) -> ColumnAnalyzer:
    """
    Construye un objeto que cumple el Protocol ColumnAnalyzer desde una función.
    """
    return type("_Analyzer", (), {"name": analysis_name, "analyze": staticmethod(fn)})()


_DEFAULT_ANALYZERS: list[ColumnAnalyzer] = [
    _make_analyzer("ks_test", ks_test),
    _make_analyzer("psi", population_stability_index),
    _make_analyzer("kl_js", kl_js_divergence),
]


def run_analysis(
    ref_path: str,
    prod_path: str,
    reader: DataReader,
    analyzers: Optional[list[ColumnAnalyzer]] = None,
    extra_analyzers: Optional[list[ColumnAnalyzer]] = None,
    columns: Optional[list[str]] = None,
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
    active_analyzers = (analyzers if analyzers is not None else _DEFAULT_ANALYZERS) + (
        extra_analyzers or []
    )

    ref_schema = reader.schema(ref_path)
    prod_schema = reader.schema(prod_path)
    schema_diff = detect_schema_drift(ref_schema, prod_schema)

    target_cols = common_numeric_columns(ref_schema, prod_schema)
    if columns is not None:
        target_cols = [c for c in target_cols if c in columns]

    ref_data = reader.read(ref_path)
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
