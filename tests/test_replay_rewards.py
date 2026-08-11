from __future__ import annotations

import json

import pytest

from evalharness.replay_rewards import load_raw_predictions, replay_rewards
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


def _row(sku_id: str, category: str, pack) -> Row:
    labels = {
        name: AttributeLabel(status=LabelStatus.UNKNOWN)
        for name in pack.specs
    }
    labels["garment_category"] = AttributeLabel(
        value=category,
        status=LabelStatus.LABELED,
    )
    return Row(
        sku_id=sku_id,
        source="test",
        split="eval",
        input=RowInput(title=sku_id),
        labels=labels,
        provenance=Provenance(labeler="test", prompt_version="test"),
    )


def _clean_record(category: str, pack) -> dict:
    record = {
        name: [pack.unknown_token] if spec.kind == "multi" else pack.unknown_token
        for name, spec in pack.specs.items()
    }
    record["garment_category"] = category
    return record


def test_replay_uses_production_components_weights_and_transitions(pack):
    gold = [_row("a", "dress", pack), _row("b", "top", pack)]
    baseline = {
        "a": json.dumps(_clean_record("dress", pack)),
        "b": "not json",
    }
    candidate = {
        "a": json.dumps(_clean_record("top", pack)),
        "b": json.dumps(_clean_record("top", pack)),
    }

    result = replay_rewards(gold, baseline, candidate, pack)

    assert result["reward_weights"] == [1.0, 1.0, 2.0]
    assert result["baseline"]["components"]["format_validity"]["passes"] == 1
    assert result["candidate"]["components"]["format_validity"]["passes"] == 2
    assert result["baseline"]["components"]["golden_agreement"]["passes"] == 1
    assert result["candidate"]["components"]["golden_agreement"]["passes"] == 1
    assert result["paired_component_transitions"]["golden_agreement"] == {
        "both_pass": 0,
        "baseline_only_pass": 1,
        "candidate_only_pass": 1,
        "both_fail": 0,
    }


def test_replay_rejects_incomplete_pairing(pack):
    gold = [_row("a", "dress", pack), _row("b", "top", pack)]
    baseline = {
        "a": json.dumps({"garment_category": "dress"}),
        "b": json.dumps({"garment_category": "top"}),
    }
    with pytest.raises(ValueError, match="candidate SKU set differs"):
        replay_rewards(gold, baseline, {"a": baseline["a"]}, pack)


def test_raw_prediction_loader_rejects_duplicates_and_non_strings(tmp_path):
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '\n'.join(
            [
                json.dumps({"sku_id": "a", "raw": "{}"}),
                json.dumps({"sku_id": "a", "raw": "{}"}),
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_raw_predictions(duplicate)

    non_string = tmp_path / "non-string.jsonl"
    non_string.write_text(json.dumps({"sku_id": "a", "raw": {}}) + "\n")
    with pytest.raises(TypeError, match="raw string"):
        load_raw_predictions(non_string)
