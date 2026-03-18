import numpy as np
import polars as pl
import pytest

from dqa.adapters.readers.polars_reader import PolarsReader
from dqa.domain.ports import DataReader


def test_implements_data_reader_protocol():
    assert isinstance(PolarsReader(), DataReader)


def test_read_parquet_returns_only_numeric_columns(reference_parquet):
    data = PolarsReader().read(reference_parquet)
    assert "age" in data
    assert "price" in data
    assert "name" not in data


def test_read_returns_numpy_arrays(reference_parquet):
    data = PolarsReader().read(reference_parquet)
    assert all(isinstance(arr, np.ndarray) for arr in data.values())


def test_schema_returns_dtype_strings(reference_parquet):
    schema = PolarsReader().schema(reference_parquet)
    assert schema["age"] in ("float64", "float32")
    assert schema["price"] in ("float64", "float32")
    assert "name" in schema


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
