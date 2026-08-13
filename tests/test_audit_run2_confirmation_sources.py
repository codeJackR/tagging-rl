from __future__ import annotations

import pytest

from training.audit_run2_confirmation_sources import audit_source_membership


def _inputs() -> dict:
    return {
        "raw_feed": {"train-a", "probe-a", "eval-a"},
        "raw_labeled": {"train-a", "probe-a", "eval-a"},
        "sft": {"train-a"},
        "probe": {"probe-a"},
        "eval_candidates": {"eval-a"},
        "frozen_eval": {"eval-a"},
    }


def test_complete_partition_finds_no_unused_labeled_rows() -> None:
    result = audit_source_membership(**_inputs())

    assert result["partition_covers_labeled_universe"] is True
    assert result["unallocated_labeled_rows"] == 0
    assert result["eval_candidates_equal_burned_frozen_eval"] is True


def test_unused_labeled_row_is_reported() -> None:
    inputs = _inputs()
    inputs["raw_feed"].add("unused")
    inputs["raw_labeled"].add("unused")

    result = audit_source_membership(**inputs)
    assert result["unallocated_labeled_skus"] == ["unused"]
    assert result["partition_covers_labeled_universe"] is False


def test_raw_source_universes_must_match() -> None:
    inputs = _inputs()
    inputs["raw_feed"].add("feed-only")

    with pytest.raises(ValueError, match="universes disagree"):
        audit_source_membership(**inputs)


def test_allocations_must_be_disjoint() -> None:
    inputs = _inputs()
    inputs["probe"].add("train-a")

    with pytest.raises(ValueError, match="allocations overlap"):
        audit_source_membership(**inputs)


def test_eval_candidates_must_equal_frozen_eval() -> None:
    inputs = _inputs()
    inputs["frozen_eval"] = {"different"}

    with pytest.raises(ValueError, match="SKU sets disagree"):
        audit_source_membership(**inputs)
