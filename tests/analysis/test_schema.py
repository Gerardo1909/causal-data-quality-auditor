from dqa.analysis.schema import common_numeric_columns, detect_schema_drift


def test_identical_schemas_no_diff():
    ref = {"age": "float64", "price": "float64"}
    prod = {"age": "float64", "price": "float64"}
    assert not detect_schema_drift(ref, prod).has_changes


def test_detects_added_column():
    diff = detect_schema_drift(
        {"age": "float64"}, {"age": "float64", "score": "float64"}
    )
    assert "score" in diff.added
    assert not diff.removed


def test_detects_removed_column():
    diff = detect_schema_drift(
        {"age": "float64", "price": "float64"}, {"age": "float64"}
    )
    assert "price" in diff.removed
    assert not diff.added


def test_detects_type_change():
    diff = detect_schema_drift({"age": "float64"}, {"age": "int32"})
    assert diff.type_changed["age"] == ("float64", "int32")


def test_common_numeric_columns_intersection():
    ref = {"age": "float64", "name": "str", "price": "float64"}
    prod = {"age": "float64", "name": "str", "score": "float64"}
    assert common_numeric_columns(ref, prod) == ["age"]


def test_common_numeric_columns_excludes_type_mismatch():
    # "age" existe en ambos pero con dtype distinto → no se incluye
    ref = {"age": "float64"}
    prod = {"age": "int32"}
    assert common_numeric_columns(ref, prod) == []
