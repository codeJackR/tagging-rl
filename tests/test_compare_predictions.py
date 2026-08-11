from __future__ import annotations

import pytest

from evalharness.compare_predictions import (
    compare_predictions,
    summarize_rule_transitions,
)
from labeling.records import (
    AttributeLabel,
    LabelStatus,
    Provenance,
    Row,
    RowInput,
)
from verifier import load_pack


@pytest.fixture(scope="module")
def pack():
    return load_pack("packs/vastraa_taste_v1")


def _row(sku_id: str, category: str) -> Row:
    return Row(
        sku_id=sku_id,
        source="test",
        split="eval",
        input=RowInput(title=sku_id),
        labels={
            "garment_category": AttributeLabel(
                value=category,
                status=LabelStatus.LABELED,
            )
        },
        provenance=Provenance(labeler="test", prompt_version="test"),
    )


def test_compare_predictions_reports_attribute_and_class_deltas(pack):
    gold = [_row("a", "dress"), _row("b", "top"), _row("c", "dress")]
    baseline = {
        "a": {"garment_category": "top"},
        "b": {"garment_category": "top"},
        "c": {"garment_category": "top"},
    }
    candidate = {
        "a": {"garment_category": "dress"},
        "b": {"garment_category": "top"},
        "c": {"garment_category": "dress"},
    }

    result = compare_predictions(
        gold,
        baseline,
        candidate,
        {sku: [] for sku in baseline},
        {sku: [] for sku in candidate},
        pack,
    )

    attribute = result["attributes"]["garment_category"]
    assert result["headline"]["delta_macro_f1"] > 0.0
    assert result["attribute_direction_counts"] == {
        "improved": 1,
        "unchanged": 0,
        "regressed": 0,
    }
    assert attribute["classes"]["dress"]["delta_f1"] > 0.0
    assert attribute["candidate"]["exact_match"] == 1.0


def test_rule_transitions_preserve_added_and_removed_skus():
    result = summarize_rule_transitions(
        ["a", "b", "c", "d"],
        {"a": [], "b": ["old"], "c": ["shared"], "d": []},
        {"a": ["new"], "b": [], "c": ["shared", "new"], "d": []},
    )

    assert result["baseline_total_violations"] == 2
    assert result["candidate_total_violations"] == 3
    assert result["row_transitions"] == {
        "baseline_only_violation": 1,
        "both_clean": 1,
        "both_have_violations": 1,
        "candidate_only_violation": 1,
    }
    assert result["rules"]["new"]["added_skus"] == ["a", "c"]
    assert result["rules"]["old"]["removed_skus"] == ["b"]


def test_compare_predictions_rejects_incomplete_pairing(pack):
    gold = [_row("a", "dress"), _row("b", "top")]
    baseline = {
        "a": {"garment_category": "dress"},
        "b": {"garment_category": "top"},
    }
    with pytest.raises(ValueError, match="candidate SKU set differs"):
        compare_predictions(
            gold,
            baseline,
            {"a": baseline["a"]},
            {"a": [], "b": []},
            {"a": []},
            pack,
        )
