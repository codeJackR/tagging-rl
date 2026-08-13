from __future__ import annotations

import json

import pytest

from evalharness.transition_analysis import (
    _load_raw_predictions,
    analyze_transitions,
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


def _row(
    sku_id: str,
    *,
    category: str = "dress",
    details: list[str] | None = None,
    category_unknown: bool = False,
) -> Row:
    labels = {
        name: AttributeLabel(status=LabelStatus.UNKNOWN)
        for name in load_pack("packs/vastraa_taste_v1").specs
    }
    labels["garment_category"] = (
        AttributeLabel(status=LabelStatus.UNKNOWN)
        if category_unknown
        else AttributeLabel(value=category, status=LabelStatus.LABELED)
    )
    if details is not None:
        labels["details"] = AttributeLabel(
            value=details, status=LabelStatus.LABELED
        )
    return Row(
        sku_id=sku_id,
        source="test",
        split="eval",
        input=RowInput(title=f"Title {sku_id}"),
        labels=labels,
        provenance=Provenance(labeler="test", prompt_version="test"),
    )


def test_all_state_transitions_and_wrong_subtypes_are_exhaustive(pack):
    cases = [
        ("aa", "unknown", "unknown", "abstain_to_abstain"),
        ("ac", "unknown", "dress", "abstain_to_correct"),
        ("aw", "unknown", "top", "abstain_to_wrong"),
        ("ca", "dress", "unknown", "correct_to_abstain"),
        ("cc", "dress", "dress", "unchanged_correct"),
        ("cw", "dress", "top", "correct_to_wrong"),
        ("wa", "top", "unknown", "wrong_to_abstain"),
        ("wc", "top", "dress", "wrong_to_correct"),
        ("ww", "top", "top", "unchanged_wrong"),
        ("wd", "top", "coat", "wrong_to_different_wrong"),
    ]
    gold = [_row(sku) for sku, *_ in cases]
    baseline = {sku: {"garment_category": before} for sku, before, _, _ in cases}
    candidate = {sku: {"garment_category": after} for sku, _, after, _ in cases}

    result = analyze_transitions(gold, baseline, candidate, pack)
    cells = {
        cell["sku_id"]: cell
        for cell in result["cells"]
        if cell["attribute"] == "garment_category"
    }

    assert {sku: cells[sku]["transition"] for sku, *_ in cases} == {
        sku: expected for sku, _, _, expected in cases
    }
    assert sum(result["overall"]["state_transition_counts"].values()) == 10
    assert result["overall"]["baseline_abstain_exits"] == {
        "total": 2,
        "to_correct": 1,
        "to_wrong": 1,
        "to_wrong_minus_to_correct": 0,
        "correct_share": 0.5,
        "wrong_share": 0.5,
    }


def test_gold_unknown_is_excluded_and_multi_value_order_is_ignored(pack):
    gold = [
        _row("multi", details=["pleated", "embroidered"]),
        _row("unknown", category_unknown=True),
    ]
    baseline = {
        "multi": {"details": ["embroidered", "pleated"]},
        "unknown": {"garment_category": "dress"},
    }
    candidate = {
        "multi": {"details": ["pleated", "embroidered"]},
        "unknown": {"garment_category": "top"},
    }

    result = analyze_transitions(gold, baseline, candidate, pack)
    detail_cells = [
        cell for cell in result["cells"] if cell["attribute"] == "details"
    ]

    assert len(detail_cells) == 1
    assert detail_cells[0]["transition"] == "unchanged_correct"
    details = result["multi_value_attributes"]["details"]
    assert details["gold_known_cells"] == 1
    assert details["common_committed_class_concentration"]["baseline"][
        "denominator"
    ] == 2
    assert result["attributes"]["garment_category"]["gold_known_cells"] == 1


def test_h1_direction_is_based_on_abstain_exit_outcomes(pack):
    gold = [_row("wrong"), _row("right"), _row("wrong2")]
    baseline = {
        sku: {"garment_category": "unknown"} for sku in ("wrong", "right", "wrong2")
    }
    candidate = {
        "wrong": {"garment_category": "top"},
        "right": {"garment_category": "dress"},
        "wrong2": {"garment_category": "coat"},
    }

    overall = analyze_transitions(gold, baseline, candidate, pack)["overall"]
    assert overall["h1_descriptive_direction"] == (
        "strengthens_overcommitment_hypothesis"
    )
    assert overall["baseline_abstain_exits"]["wrong_share"] == pytest.approx(2 / 3)


def test_pairing_and_raw_loader_fail_closed(pack, tmp_path):
    gold = [_row("a"), _row("b")]
    with pytest.raises(ValueError, match="candidate SKU set differs"):
        analyze_transitions(
            gold,
            {"a": {}, "b": {}},
            {"a": {}},
            pack,
        )

    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"sku_id": "a", "raw": '{"garment_category":"dress"}'}),
                json.dumps({"sku_id": "a", "raw": '{"garment_category":"top"}'}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prediction SKU ID"):
        _load_raw_predictions(path, pack)
