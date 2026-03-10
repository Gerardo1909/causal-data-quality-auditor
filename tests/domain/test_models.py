import pytest

from dqa.domain.models import AnalysisResult, ColumnReport, DatasetReport, DriftLevel, SchemaDiff


def test_drift_level_ordering():
    assert DriftLevel.STABLE < DriftLevel.WARNING < DriftLevel.ALERT


def test_drift_level_str():
    assert str(DriftLevel.STABLE) == "STABLE"
    assert str(DriftLevel.ALERT) == "ALERT"


def test_column_report_worst_level_no_results():
    report = ColumnReport(name="price", dtype="float64", results={})
    assert report.worst_level == DriftLevel.STABLE


def test_column_report_worst_level_picks_max():
    report = ColumnReport(
        name="price",
        dtype="float64",
        results={
            "ks_test": AnalysisResult(level=DriftLevel.STABLE, details={}),
            "psi": AnalysisResult(level=DriftLevel.ALERT, details={}),
        },
    )
    assert report.worst_level == DriftLevel.ALERT


def test_dataset_report_has_alerts_true():
    col_ok = ColumnReport(
        name="age",
        dtype="float64",
        results={"psi": AnalysisResult(level=DriftLevel.STABLE, details={})},
    )
    col_alert = ColumnReport(
        name="price",
        dtype="float64",
        results={"psi": AnalysisResult(level=DriftLevel.ALERT, details={})},
    )
    report = DatasetReport(columns=[col_ok, col_alert])
    assert report.has_alerts is True
    assert report.alert_columns == ["price"]


def test_dataset_report_has_alerts_false():
    col = ColumnReport(
        name="age",
        dtype="float64",
        results={"psi": AnalysisResult(level=DriftLevel.STABLE, details={})},
    )
    report = DatasetReport(columns=[col])
    assert report.has_alerts is False
    assert report.alert_columns == []


def test_dataset_report_overall_level():
    col = ColumnReport(
        name="x",
        dtype="float64",
        results={"psi": AnalysisResult(level=DriftLevel.WARNING, details={})},
    )
    assert DatasetReport(columns=[col]).overall_level == DriftLevel.WARNING


def test_dataset_report_overall_level_empty():
    assert DatasetReport(columns=[]).overall_level == DriftLevel.STABLE


def test_schema_diff_has_changes_added():
    assert SchemaDiff(added=["col_new"]).has_changes is True


def test_schema_diff_has_changes_removed():
    assert SchemaDiff(removed=["col_old"]).has_changes is True


def test_schema_diff_has_changes_type():
    assert SchemaDiff(type_changed={"age": ("float64", "int32")}).has_changes is True


def test_schema_diff_no_changes():
    assert SchemaDiff().has_changes is False
