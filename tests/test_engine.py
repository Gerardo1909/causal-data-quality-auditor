import numpy as np
import pytest

from dqa.domain.models import DatasetReport, DriftLevel
from dqa.engine import run_analysis

RNG = np.random.default_rng(42)


class FakeReader:
    """Mock de DataReader que sirve datos en memoria sin I/O."""

    def __init__(self, ref_data: dict, prod_data: dict, schema: dict | None = None):
        self._data = {"ref": ref_data, "prod": prod_data}
        self._schema = schema or {k: "float64" for k in ref_data}

    def read(self, path: str) -> dict:
        return self._data[path]

    def schema(self, path: str) -> dict:
        return self._schema


def make_reader(ref: dict, prod: dict, schema: dict | None = None) -> FakeReader:
    return FakeReader(ref, prod, schema)


def test_returns_dataset_report():
    ref = {"price": RNG.normal(0, 1, 200)}
    prod = {"price": RNG.normal(0, 1, 200)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod))
    assert isinstance(report, DatasetReport)


def test_analyzes_all_common_numeric_columns():
    ref = {"price": RNG.normal(0, 1, 200), "age": RNG.normal(30, 5, 200)}
    prod = {"price": RNG.normal(0, 1, 200), "age": RNG.normal(30, 5, 200)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod))
    names = {c.name for c in report.columns}
    assert "price" in names and "age" in names


def test_filters_to_requested_columns():
    ref = {"price": RNG.normal(0, 1, 200), "age": RNG.normal(30, 5, 200)}
    prod = {"price": RNG.normal(0, 1, 200), "age": RNG.normal(30, 5, 200)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod), columns=["price"])
    assert len(report.columns) == 1
    assert report.columns[0].name == "price"


def test_detects_drift_on_drifted_column():
    ref = {"price": RNG.normal(0, 1, 500)}
    prod = {"price": RNG.normal(10, 1, 500)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod))
    price = next(c for c in report.columns if c.name == "price")
    assert price.worst_level == DriftLevel.ALERT


def test_schema_diff_included_in_report():
    ref_schema = {"price": "float64", "age": "float64"}
    prod_schema = {"price": "float64"}

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
    ref = {"x": RNG.normal(0, 1, 200)}
    prod = {"x": RNG.normal(0, 1, 200)}
    report = run_analysis("ref", "prod", reader=make_reader(ref, prod))
    assert len(report.columns) == 1
    keys = set(report.columns[0].results.keys())
    assert {"ks_test", "psi", "kl_js"} == keys
