import os

from typer.testing import CliRunner

from dqa.cli.main import app

runner = CliRunner()


def test_help_shows_compare_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "compare" in result.output


def test_compare_stable_exits_0(reference_parquet):
    """
    Comparar un dataset consigo mismo no debe disparar drift.
    """
    result = runner.invoke(app, ["compare", reference_parquet, reference_parquet])
    assert result.exit_code == 0


def test_compare_drifted_exits_1(reference_parquet, production_parquet):
    """
    price tiene drift de ~2.5σ — debe disparar alert y exit 1.
    """
    result = runner.invoke(
        app,
        [
            "compare",
            reference_parquet,
            production_parquet,
            "--fail-on",
            "alert",
        ],
    )
    assert result.exit_code == 1


def test_fail_on_never_always_exits_0(reference_parquet, production_parquet):
    result = runner.invoke(
        app,
        [
            "compare",
            reference_parquet,
            production_parquet,
            "--fail-on",
            "never",
        ],
    )
    assert result.exit_code == 0


def test_compare_unsupported_format_exits_2(tmp_path):
    path = str(tmp_path / "file.xlsx")
    open(path, "w").close()
    result = runner.invoke(app, ["compare", path, path])
    assert result.exit_code == 2


def test_compare_markdown_creates_file(reference_parquet, tmp_path):
    out = str(tmp_path / "report.md")
    result = runner.invoke(
        app,
        [
            "compare",
            reference_parquet,
            reference_parquet,
            "--format",
            "markdown",
            "--output",
            out,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(out)
    with open(out) as f:
        assert "DQA Drift Report" in f.read()


def test_compare_specific_columns(reference_parquet):
    result = runner.invoke(
        app,
        [
            "compare",
            reference_parquet,
            reference_parquet,
            "--columns",
            "age",
        ],
    )
    assert result.exit_code == 0
