from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

from labeling.records import AttributeLabel, LabelStatus, read_jsonl
from training.build_cb_class_weights import (
    DEFAULT_OUTPUT,
    NOT_APPLICABLE_CLASS,
    VERSION,
    build_cb_class_weight_artifact,
    derive_class_weight_map,
    label_class_keys,
)
from training.reward_scale_contract import CLASS_WEIGHT_MAX, CLASS_WEIGHT_MIN, class_weight
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def test_label_support_keys_exclude_unknown_and_include_not_applicable(pack):
    assert label_class_keys(
        field_name="occasion",
        label=AttributeLabel(status=LabelStatus.UNKNOWN),
        pack=pack,
    ) == ()
    assert label_class_keys(
        field_name="occasion",
        label=AttributeLabel(status=LabelStatus.NOT_APPLICABLE),
        pack=pack,
    ) == (NOT_APPLICABLE_CLASS,)
    assert label_class_keys(
        field_name="occasion",
        label=AttributeLabel(status=LabelStatus.LABELED, value="work"),
        pack=pack,
    ) == ("work",)
    assert label_class_keys(
        field_name="details",
        label=AttributeLabel(
            status=LabelStatus.LABELED, value=["lined", "washed"]
        ),
        pack=pack,
    ) == ("lined", "washed")


def test_real_map_has_locked_counts_bounds_and_explicit_na(pack):
    rows = read_jsonl(ROOT / "data" / "train_weak_grpo_cap4_sft_train_v1.jsonl")
    result = derive_class_weight_map(rows, pack)

    assert result["rows"] == 1_438
    assert result["fields"] == 15
    assert result["known_field_cells"] == 12_533
    assert result["unknown_field_cells"] == 9_037
    assert result["observed_attribute_class_pairs"] == 116
    assert result["classes_below_five"] == 17
    assert result["classes_below_ten"] == 30
    assert result["class_observations"] == (
        result["known_field_cells"] + result["multi_label_extra_observations"]
    )
    assert result["multi_label_extra_observations"] == 38
    assert result["minimum_derived_weight"] >= CLASS_WEIGHT_MIN
    assert result["maximum_derived_weight"] <= CLASS_WEIGHT_MAX
    assert result["clipping_counts"] == {
        "minimum": 24,
        "maximum": 20,
        "unclipped": 72,
    }
    assert sum(result["clipping_counts"].values()) == 116
    assert result["not_applicable_fields"]
    for attribute in result["attributes"].values():
        assert "unknown" not in attribute["classes"]


def test_every_real_weight_recomputes_from_its_attribute_median(pack):
    rows = read_jsonl(ROOT / "data" / "train_weak_grpo_cap4_sft_train_v1.jsonl")
    result = derive_class_weight_map(rows, pack)

    for attribute in result["attributes"].values():
        supports = [entry["support"] for entry in attribute["classes"].values()]
        assert attribute["median_positive_support"] == statistics.median(supports)
        for entry in attribute["classes"].values():
            expected_raw = math.sqrt(
                attribute["median_positive_support"] / entry["support"]
            )
            assert entry["raw_weight"] == expected_raw
            assert entry["weight"] == class_weight(
                entry["support"], attribute["median_positive_support"]
            )


def test_artifact_rebuild_is_deterministic_and_matches_published_file():
    first = build_cb_class_weight_artifact(repo_root=ROOT)
    second = build_cb_class_weight_artifact(repo_root=ROOT)
    published = json.loads((ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))

    assert first == second == published
    assert first["version"] == VERSION
    assert first["status"] == "passed"
    assert first["selection_boundary"]["candidate_completion_rewards_calculated"] is False
    assert all(first["invariants"].values())
    assert first["cuda_imports_performed"] is False


def test_derivation_fails_closed_on_empty_duplicate_or_incomplete_rows(pack):
    rows = read_jsonl(ROOT / "data" / "train_weak_grpo_cap4_sft_train_v1.jsonl")
    with pytest.raises(ValueError, match="cannot be empty"):
        derive_class_weight_map([], pack)
    with pytest.raises(ValueError, match="duplicate SKUs"):
        derive_class_weight_map([rows[0], rows[0]], pack)

    incomplete = rows[0].model_copy(deep=True)
    incomplete.labels.pop("occasion")
    with pytest.raises(ValueError, match="labels differ from pack fields"):
        derive_class_weight_map([incomplete], pack)
