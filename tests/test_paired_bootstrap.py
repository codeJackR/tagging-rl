from __future__ import annotations

import pytest

from evalharness.paired_bootstrap import linear_percentile, paired_bootstrap
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


def test_linear_percentile_interpolates_and_checks_inputs():
    assert linear_percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    assert linear_percentile([3.0], 0.95) == 3.0
    with pytest.raises(ValueError, match="at least one"):
        linear_percentile([], 0.5)
    with pytest.raises(ValueError, match="between zero and one"):
        linear_percentile([1.0], 1.1)


def test_paired_bootstrap_is_deterministic_and_tracks_candidate_minus_baseline(pack):
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

    first = paired_bootstrap(
        gold, baseline, candidate, pack, seed=17, replicates=100
    )
    second = paired_bootstrap(
        gold, baseline, candidate, pack, seed=17, replicates=100
    )

    assert first == second
    macro = first["metrics"]["macro_f1"]
    assert macro["candidate"]["point"] == pytest.approx(1.0)
    assert macro["delta_candidate_minus_baseline"]["point"] > 0.0
    assert macro["delta_candidate_minus_baseline"]["fraction_below_zero"] == 0.0
    assert len(first["replicate_stream_sha256"]) == 64


def test_paired_bootstrap_rejects_unpaired_or_duplicate_inputs(pack):
    gold = [_row("a", "dress"), _row("b", "top")]
    complete = {
        "a": {"garment_category": "dress"},
        "b": {"garment_category": "top"},
    }

    with pytest.raises(ValueError, match="candidate SKU set differs"):
        paired_bootstrap(
            gold,
            complete,
            {"a": complete["a"]},
            pack,
            seed=1,
            replicates=2,
        )
    with pytest.raises(ValueError, match="duplicate SKU"):
        paired_bootstrap(
            [gold[0], gold[0]],
            {"a": complete["a"]},
            {"a": complete["a"]},
            pack,
            seed=1,
            replicates=2,
        )


def test_paired_bootstrap_rejects_invalid_run_settings(pack):
    gold = [_row("a", "dress")]
    predictions = {"a": {"garment_category": "dress"}}

    with pytest.raises(ValueError, match="replicates must be positive"):
        paired_bootstrap(
            gold, predictions, predictions, pack, seed=1, replicates=0
        )
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap(
            gold,
            predictions,
            predictions,
            pack,
            seed=1,
            replicates=1,
            confidence=1.0,
        )
