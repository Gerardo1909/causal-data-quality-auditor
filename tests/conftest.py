"""
Fixtures compartidos para la suite de tests.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="session")
def reference_parquet(tmp_path_factory):
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "age": rng.normal(35, 10, 1000).tolist(),
            "price": rng.normal(100, 20, 1000).tolist(),
            "name": ["Alice"] * 500 + ["Bob"] * 500,
        }
    )
    path = str(tmp_path_factory.mktemp("fixtures") / "reference.parquet")
    df.write_parquet(path)
    return path


@pytest.fixture(scope="session")
def production_parquet(tmp_path_factory):
    rng = np.random.default_rng(99)
    df = pl.DataFrame(
        {
            "age": rng.normal(35, 10, 1000).tolist(),
            "price": rng.normal(150, 20, 1000).tolist(),
            "name": ["Alice"] * 1000,
        }
    )
    path = str(tmp_path_factory.mktemp("fixtures") / "production.parquet")
    df.write_parquet(path)
    return path
