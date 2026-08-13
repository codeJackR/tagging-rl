from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from training.verify_run2_d3_independent import (
    _assert_same,
    _publish_exclusive,
    _shape,
)


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_imports_only_python_standard_library():
    tree = ast.parse(
        (ROOT / "training/verify_run2_d3_independent.py").read_text(encoding="utf-8")
    )
    project_roots = {"training", "tests", "labeling", "evalharness", "verifier"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & project_roots


def test_group_shape_uses_canonical_ties_and_28_pair_denominator():
    shape = _shape([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    assert shape["unique"] == 5
    assert shape["largest_tie"] == 2
    assert shape["discrimination"] == 25 / 28

    rounded = _shape([0.0, 1e-13, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert rounded["unique"] == 2
    assert rounded["largest_tie"] == 6


def test_recursive_comparison_fails_on_numeric_or_denominator_drift():
    expected = {"groups": 2, "values": [0.1, {"share": 0.5}]}
    _assert_same(expected, expected, "fixture")
    with pytest.raises(ValueError, match="numeric value differs"):
        _assert_same(
            {"groups": 2, "values": [0.2, {"share": 0.5}]},
            expected,
            "fixture",
        )
    with pytest.raises(ValueError, match="differs"):
        _assert_same(
            {"groups": 3, "values": [0.1, {"share": 0.5}]},
            expected,
            "fixture",
        )


def test_independent_report_publication_is_exclusive(tmp_path):
    output = tmp_path / "verification.json"
    value = {"status": "synthetic verification"}
    _publish_exclusive(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="output already exists"):
        _publish_exclusive(output, value)
    assert output.read_bytes() == original
