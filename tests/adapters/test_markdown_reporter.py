import io

from dqa.adapters.reporters.markdown_reporter import MarkdownReporter
from dqa.domain.models import AnalysisResult, ColumnReport, DatasetReport, DriftLevel, SchemaDiff


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
