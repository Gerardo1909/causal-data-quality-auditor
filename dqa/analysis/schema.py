"""
Primera capa de detección de schema drift. Sin dependencias externas.
"""

from __future__ import annotations

from dqa.domain.models import SchemaDiff

_NUMERIC_DTYPES = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    }
)


def detect_schema_drift(ref: dict[str, str], prod: dict[str, str]) -> SchemaDiff:
    """
    Compara dos schemas (col → dtype_str) y retorna sus diferencias estructurales.

    Args:
        ref:  Schema del dataset de referencia {columna: dtype}.
        prod: Schema del dataset de producción {columna: dtype}.

    Returns:
        SchemaDiff con columnas agregadas, removidas y con tipo cambiado.
    """
    ref_cols = set(ref)
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
        if ref[col] in _NUMERIC_DTYPES and ref[col] == prod[col]
    )
