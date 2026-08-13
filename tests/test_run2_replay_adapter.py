from __future__ import annotations

from copy import deepcopy

import pytest

from training.reward_scale_contract import KNOWN_MIX_WEIGHT, UNKNOWN_MIX_WEIGHT
from training.run2_replay_adapter import (
    VERSION,
    adapt_replay_group,
    build_class_support_lookup,
    class_support_band,
    difficulty_band,
)


KNOWN_FIELDS = ("garment_category", "fit")
UNKNOWN_FIELDS = tuple(f"unknown_field_{index}" for index in range(13))


def _class_artifact():
    return {
        "version": "grpo-run2-cb-class-weights-v1",
        "weight_map": {
            "observed_attribute_class_pairs": 2,
            "attributes": {
                "garment_category": {
                    "classes": {"dress": {"support": 3, "weight": 2.0}}
                },
                "fit": {
                    "classes": {"regular": {"support": 80, "weight": 0.5}}
                },
            },
        },
    }


def _known_outcomes(first_utility: float, second_utility: float, *, cb: bool):
    base = [
        {
            "field_name": "garment_category",
            "outcome": "correct" if first_utility == 1 else "wrong",
            "utility": first_utility,
            "set_f1": None,
        },
        {
            "field_name": "fit",
            "outcome": "correct" if second_utility == 1 else "abstain",
            "utility": second_utility,
            "set_f1": None,
        },
    ]
    if not cb:
        return base
    return [
        {
            **base[0],
            "gold_class_keys": ["dress"],
            "gold_class_weights": [2.0],
            "field_weight": 2.0,
        },
        {
            **base[1],
            "gold_class_keys": ["regular"],
            "gold_class_weights": [0.5],
            "field_weight": 0.5,
        },
    ]


def _unknown_outcomes(correct_count: int):
    return [
        {
            "field_name": field_name,
            "outcome": "abstain" if index < correct_count else "commit",
            "utility": 1.0 if index < correct_count else -1.0,
        }
        for index, field_name in enumerate(UNKNOWN_FIELDS)
    ]


def _completion(index: int, *, malformed: bool = False):
    first_utility = 1.0 if index >= 4 else -1.0
    second_utility = 1.0 if index % 2 else 0.0
    correct_unknown = index + 5
    known_semantic = (first_utility + second_utility) / 2
    unknown_semantic = (correct_unknown - (13 - correct_unknown)) / 13
    ua_semantic = (
        KNOWN_MIX_WEIGHT * known_semantic
        + UNKNOWN_MIX_WEIGHT * unknown_semantic
    )
    cb_known = (first_utility * 2.0 + second_utility * 0.5) / 2.5
    cb_semantic = (
        KNOWN_MIX_WEIGHT * cb_known
        + UNKNOWN_MIX_WEIGHT * unknown_semantic
    )
    source = {
        "sku_id": "synthetic-sku",
        "rollout_index": index,
        "passed": index >= 4,
        "scorable_labels": 2,
        "correct_labels": int(first_utility == 1) + int(second_utility == 1),
        "false_abstention_labels": ["fit"] if second_utility == 0 else [],
        "excluded_gold_unknown_labels": list(UNKNOWN_FIELDS),
        "correct_abstention_labels": list(UNKNOWN_FIELDS[:correct_unknown]),
        "rule_violations": [],
    }
    if malformed:
        candidates = {
            "U": {
                "candidate": "U", "reward": -1.25, "eligible": False,
                "gate_errors": ["synthetic malformed"], "rule_violations": [],
                "rule_adjustment": None, "known_semantics": None,
            },
            "UA": {
                "candidate": "UA", "reward": -1.25, "eligible": False,
                "gate_errors": ["synthetic malformed"], "rule_violations": [],
                "rule_adjustment": None, "unknown_aware_semantics": None,
            },
            "CB": {
                "candidate": "CB", "reward": -1.25, "eligible": False,
                "gate_errors": ["synthetic malformed"], "rule_violations": [],
                "rule_adjustment": None, "unknown_aware_semantics": None,
            },
        }
    else:
        known_u = {
            "semantic_score": known_semantic,
            "scorable_fields": 2,
            "excluded_gold_unknown_fields": list(UNKNOWN_FIELDS),
            "field_outcomes": _known_outcomes(first_utility, second_utility, cb=False),
        }
        unknown = {
            "semantic_score": unknown_semantic,
            "scorable_fields": 13,
            "excluded_gold_known_fields": list(KNOWN_FIELDS),
            "field_outcomes": _unknown_outcomes(correct_unknown),
        }
        candidates = {
            "U": {
                "candidate": "U", "reward": known_semantic, "eligible": True,
                "gate_errors": [], "rule_violations": [], "rule_adjustment": 0.0,
                "known_semantics": known_u,
            },
            "UA": {
                "candidate": "UA", "reward": ua_semantic, "eligible": True,
                "gate_errors": [], "rule_violations": [], "rule_adjustment": 0.0,
                "unknown_aware_semantics": {
                    "semantic_score": ua_semantic,
                    "known_semantics": known_u,
                    "unknown_semantics": unknown,
                },
            },
            "CB": {
                "candidate": "CB", "reward": cb_semantic, "eligible": True,
                "gate_errors": [], "rule_violations": [], "rule_adjustment": 0.0,
                "unknown_aware_semantics": {
                    "semantic_score": cb_semantic,
                    "known_semantics": {
                        "semantic_score": cb_known,
                        "scorable_fields": 2,
                        "total_field_weight": 2.5,
                        "excluded_gold_unknown_fields": list(UNKNOWN_FIELDS),
                        "field_outcomes": _known_outcomes(
                            first_utility, second_utility, cb=True
                        ),
                    },
                    "unknown_semantics": unknown,
                },
            },
        }
    return {
        "rollout_index": index,
        "source_rollout": source,
        "original_reward": {"weighted_total": 4.0 if index >= 4 else 2.0},
        "candidates": candidates,
    }


def _group(*, malformed_index: int | None = None):
    return {
        "group_position": 0,
        "sku_id": "synthetic-sku",
        "difficulty_sft_pass_rate": 0.5,
        "gold_record": {"garment_category": "dress", "fit": "regular"},
        "gold_known_fields": 2,
        "gold_unknown_fields": 13,
        "completions": [
            _completion(index, malformed=index == malformed_index) for index in range(8)
        ],
    }


def test_support_and_difficulty_bands_are_locked_at_boundaries():
    assert [class_support_band(value) for value in (1, 4, 5, 9, 10, 49, 50)] == [
        "rare_1_4", "rare_1_4", "low_5_9", "low_5_9",
        "medium_10_49", "medium_10_49", "common_50_plus",
    ]
    assert [difficulty_band(value / 8) for value in range(9)] == [
        "always_failed",
        "low_mixed_1_2_of_8", "low_mixed_1_2_of_8",
        "middle_mixed_3_5_of_8", "middle_mixed_3_5_of_8",
        "middle_mixed_3_5_of_8",
        "high_mixed_6_7_of_8", "high_mixed_6_7_of_8",
        "always_passed",
    ]
    with pytest.raises(ValueError, match="multiple of 1/8"):
        difficulty_band(0.3)


def test_adapter_preserves_group_identity_targets_and_segment_keys():
    lookup = build_class_support_lookup(_class_artifact())
    adapted = adapt_replay_group(_group(), class_supports=lookup)

    assert adapted.version == VERSION
    assert adapted.group_position == 0
    assert adapted.observation.group_id == "synthetic-sku"
    assert tuple(adapted.observation.rewards) == ("original", "U", "UA", "CB")
    assert all(len(values) == 8 for values in adapted.observation.rewards.values())
    assert adapted.observation.targets["known_coverage"][0] == 0.5
    assert adapted.observation.targets["selective_correctness"][0] == 0.0
    assert adapted.observation.targets["unknown_abstention_rate"][0] == 5 / 13
    assert adapted.segments.product_category == "dress"
    assert adapted.segments.difficulty_band == "middle_mixed_3_5_of_8"
    assert adapted.segments.gold_known_field_count == 2
    assert adapted.boundary["real_candidate_replay_opened"] is False


def test_null_difficulty_reconstructs_from_source_rollouts():
    group = _group()
    group["difficulty_sft_pass_rate"] = None
    adapted = adapt_replay_group(
        group, class_supports=build_class_support_lookup(_class_artifact())
    )
    assert adapted.segments.difficulty_band == "middle_mixed_3_5_of_8"


def test_recorded_difficulty_must_match_boolean_source_rollouts():
    lookup = build_class_support_lookup(_class_artifact())
    mismatch = _group()
    mismatch["difficulty_sft_pass_rate"] = 0.375
    with pytest.raises(ValueError, match="differs from source rollouts"):
        adapt_replay_group(mismatch, class_supports=lookup)

    invalid = _group()
    invalid["difficulty_sft_pass_rate"] = None
    invalid["completions"][0]["source_rollout"]["passed"] = 1
    with pytest.raises(TypeError, match="passed must be boolean"):
        adapt_replay_group(invalid, class_supports=lookup)


def test_field_and_class_children_reconstruct_semantics_without_join_explosion():
    adapted = adapt_replay_group(
        _group(), class_supports=build_class_support_lookup(_class_artifact())
    )
    first_fields = [row for row in adapted.field_contributions if row.rollout_index == 0]
    # U: 2 known. UA and CB: 2 known + 13 unknown each.
    assert len(first_fields) == 2 + 15 + 15
    totals = {
        candidate: sum(
            row.signed_contribution
            for row in first_fields
            if row.candidate == candidate
        )
        for candidate in ("U", "UA", "CB")
    }
    assert totals["U"] == pytest.approx(
        adapted.observation.targets["canonical_known_utility"][0]
    )
    assert totals["UA"] == pytest.approx(adapted.observation.rewards["UA"][0])
    assert totals["CB"] == pytest.approx(adapted.observation.rewards["CB"][0])

    cb_known_contribution = sum(
        row.signed_contribution
        for row in first_fields
        if row.candidate == "CB" and row.component == "known"
    )
    assert cb_known_contribution / KNOWN_MIX_WEIGHT == pytest.approx(
        adapted.observation.targets["class_balanced_known_utility"][0]
    )

    first_classes = [row for row in adapted.class_contributions if row.rollout_index == 0]
    assert len(first_classes) == 2
    assert {(row.field_name, row.class_support_band) for row in first_classes} == {
        ("garment_category", "rare_1_4"),
        ("fit", "common_50_plus"),
    }
    assert sum(row.absolute_contribution for row in first_classes) == pytest.approx(
        sum(
            row.absolute_contribution
            for row in first_fields
            if row.candidate == "CB" and row.component == "known"
        )
    )


def test_class_balanced_target_is_known_only_not_combined_cb_semantics():
    group = _group()
    adapted = adapt_replay_group(
        group, class_supports=build_class_support_lookup(_class_artifact())
    )
    completion = group["completions"][0]["candidates"]["CB"]
    semantics = completion["unknown_aware_semantics"]
    target = adapted.observation.targets["class_balanced_known_utility"][0]

    assert target == pytest.approx(semantics["known_semantics"]["semantic_score"])
    assert target != pytest.approx(semantics["semantic_score"])


def test_malformed_completion_has_missing_semantic_targets_and_no_children():
    adapted = adapt_replay_group(
        _group(malformed_index=3),
        class_supports=build_class_support_lookup(_class_artifact()),
    )
    assert adapted.observation.targets["canonical_known_utility"][3] is None
    assert adapted.observation.targets["class_balanced_known_utility"][3] is None
    assert adapted.observation.rewards["U"][3] == -1.25
    assert not [row for row in adapted.field_contributions if row.rollout_index == 3]
    assert not [row for row in adapted.class_contributions if row.rollout_index == 3]


def test_adapter_fails_closed_on_denominator_order_weight_and_ledger_drift():
    lookup = build_class_support_lookup(_class_artifact())

    bad_denominator = _group()
    bad_denominator["completions"][0]["source_rollout"]["scorable_labels"] = 3
    with pytest.raises(ValueError, match="denominator drifted"):
        adapt_replay_group(bad_denominator, class_supports=lookup)

    bad_order = _group()
    bad_order["completions"] = list(reversed(bad_order["completions"]))
    with pytest.raises(ValueError, match="ordered 0 through 7"):
        adapt_replay_group(bad_order, class_supports=lookup)

    bad_weight = _group()
    bad_weight["completions"][0]["candidates"]["CB"]["unknown_aware_semantics"][
        "known_semantics"
    ]["field_outcomes"][0]["gold_class_weights"][0] = 1.5
    with pytest.raises(ValueError, match="field weight is not the class-weight mean"):
        adapt_replay_group(bad_weight, class_supports=lookup)

    bad_utility = _group()
    bad_utility["completions"][0]["candidates"]["UA"]["unknown_aware_semantics"][
        "known_semantics"
    ]["field_outcomes"][0]["utility"] = 0.0
    with pytest.raises(ValueError, match="utility ledger drifted"):
        adapt_replay_group(bad_utility, class_supports=lookup)

    bad_saved_count = _group()
    bad_saved_count["completions"][0]["candidates"]["U"]["known_semantics"][
        "scorable_fields"
    ] = 3
    with pytest.raises(ValueError, match="saved scorable-field denominator drifted"):
        adapt_replay_group(bad_saved_count, class_supports=lookup)

    duplicate_unknown = _group()
    duplicate_unknown["completions"][0]["source_rollout"][
        "excluded_gold_unknown_labels"
    ][-1] = UNKNOWN_FIELDS[0]
    with pytest.raises(ValueError, match="duplicate gold-unknown field"):
        adapt_replay_group(duplicate_unknown, class_supports=lookup)

    missing_support = dict(lookup)
    del missing_support[("fit", "regular")]
    with pytest.raises(ValueError, match="missing class support"):
        adapt_replay_group(_group(), class_supports=missing_support)


def test_synthetic_adapter_output_feeds_the_analyzer_core_without_real_files():
    from training.analyze_run2_candidates import analyze_group_observations

    first = adapt_replay_group(
        _group(), class_supports=build_class_support_lookup(_class_artifact())
    )
    second_group = deepcopy(_group())
    second_group["group_position"] = 1
    second_group["sku_id"] = "synthetic-sku-2"
    for completion in second_group["completions"]:
        completion["source_rollout"]["sku_id"] = "synthetic-sku-2"
    second = adapt_replay_group(
        second_group, class_supports=build_class_support_lookup(_class_artifact())
    )
    result = analyze_group_observations(
        [first.observation, second.observation], bootstrap_replicates=20
    )
    assert result["groups"] == 2
    assert result["completions"] == 16
    assert result["boundary"]["real_candidate_replay_opened"] is False
