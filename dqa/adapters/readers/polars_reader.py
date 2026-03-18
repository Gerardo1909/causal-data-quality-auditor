"""
Adapter DataReader implementado con Polars. Soporta Parquet, CSV y NDJSON.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

_NUMERIC_POLARS_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
)


class PolarsReader:
    """
    Implementa DataReader usando Polars para máxima performance en formatos columnar.
    """

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
        """
        Retorna solo las columnas numéricas del dataset como arrays numpy.
        """
        df = self._load(path)
        return {
            col: df[col].to_numpy()
            for col in df.columns
            if df[col].dtype in _NUMERIC_POLARS_DTYPES
        }

    def schema(self, path: str) -> dict[str, str]:
        """
        ∫Retorna el schema completo (todas las columnas) como {col: dtype_str}.
        """
        df = self._load(path)
        return {col: str(df[col].dtype).lower() for col in df.columns}
