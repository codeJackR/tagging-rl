from __future__ import annotations

import pytest

from training.build_run2_data_roles import build_development_views


def _inputs() -> dict:
    order = [f"dev-{index}" for index in range(9)]
    return {
        "validation_order": order,
        "validation_families": {sku: f"family-{sku}" for sku in order},
        "training_skus": {"train-1"},
        "training_families": {"family-train"},
        "pass_counts": {sku: index for index, sku in enumerate(order)},
    }


def test_views_partition_development_by_preoutcome_difficulty() -> None:
    result = build_development_views(**_inputs())

    assert result["source_rows"] == 9
    assert result["views"]["difficult_0_to_2_of_8"]["rows"] == 3
    assert result["views"]["middle_3_to_5_of_8"]["rows"] == 3
    assert result["views"]["easy_retention_6_to_8_of_8"]["rows"] == 3
    assert result["invariants"]["difficulty_views_partition_source_exactly_once"] is True


def test_sku_overlap_fails_closed() -> None:
    inputs = _inputs()
    inputs["training_skus"] = {"dev-0"}

    with pytest.raises(ValueError, match="SKU sets overlap"):
        build_development_views(**inputs)


def test_family_overlap_fails_closed() -> None:
    inputs = _inputs()
    inputs["training_families"] = {"family-dev-0"}

    with pytest.raises(ValueError, match="family sets overlap"):
        build_development_views(**inputs)


def test_missing_difficulty_row_fails_closed() -> None:
    inputs = _inputs()
    inputs["pass_counts"].pop("dev-0")

    with pytest.raises(ValueError, match="membership must match exactly"):
        build_development_views(**inputs)


def test_invalid_pass_count_fails_closed() -> None:
    inputs = _inputs()
    inputs["pass_counts"]["dev-0"] = 9

    with pytest.raises(ValueError, match="zero through eight"):
        build_development_views(**inputs)
