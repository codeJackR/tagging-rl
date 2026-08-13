from __future__ import annotations

import json
import math

import pytest

from training.run2_comparison_contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPARISON_DECIMALS,
    EXPECTED_ACTIVE_COMPLETIONS,
    EXPECTED_ACTIVE_GROUPS,
    PAIR_COUNT_PER_GROUP,
    VERSION,
    build_contract,
    canonical_number,
    directional_net_alignment,
    group_reward_shape,
    harmful_coverage_preference,
)


def test_group_shape_counts_ties_and_is_invariant_to_positive_affine_scale():
    rewards = [0, 0, 0, 1, 1, 2, 3, 4]
    shape = group_reward_shape(rewards)
    transformed = group_reward_shape([10 + 7 * value for value in rewards])

    assert PAIR_COUNT_PER_GROUP == 28
    assert shape["unique_reward_levels"] == 5
    assert shape["largest_tie_size"] == 3
    assert shape["tied_pairs"] == 4
    assert shape["discriminated_pairs"] == 24
    assert shape["pairwise_discrimination_rate"] == 24 / 28
    for key in (
        "unique_reward_levels",
        "largest_tie_size",
        "tied_pairs",
        "discriminated_pairs",
        "pairwise_discrimination_rate",
    ):
        assert transformed[key] == shape[key]
    assert transformed["population_variance_own_scale"] != shape["population_variance_own_scale"]


def test_canonicalization_removes_float_dust_but_preserves_real_differences():
    assert COMPARISON_DECIMALS == 12
    assert canonical_number(-0.0) == 0.0
    shape = group_reward_shape([0.0, 4e-14, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert shape["unique_reward_levels"] == 6
    with pytest.raises(ValueError, match="finite"):
        canonical_number(math.inf)


def test_directional_alignment_keeps_reward_ties_in_denominator():
    target = list(range(8))
    perfect = directional_net_alignment(target, target)
    reverse = directional_net_alignment(list(reversed(target)), target)
    flat = directional_net_alignment([0] * 8, target)

    assert perfect == {
        "comparable_pairs": 28,
        "concordant_pairs": 28,
        "discordant_pairs": 0,
        "reward_ties": 0,
        "net_alignment": 1.0,
    }
    assert reverse["net_alignment"] == -1.0
    assert flat["reward_ties"] == 28
    assert flat["net_alignment"] == 0.0


def test_harmful_coverage_metric_penalizes_gaming_and_half_penalizes_ties():
    coverage = list(range(8))
    utility = list(reversed(range(8)))
    harmful = harmful_coverage_preference(coverage, coverage, utility)
    safe = harmful_coverage_preference(utility, coverage, utility)
    flat = harmful_coverage_preference([0] * 8, coverage, utility)

    assert harmful["harmful_preference_score"] == 1.0
    assert safe["harmful_preference_score"] == 0.0
    assert flat["harmful_preference_score"] == 0.5


def test_contract_is_locked_to_manifest_without_opening_candidate_records():
    contract = build_contract(repo_root=".")

    assert contract["version"] == VERSION
    assert contract["status"] == "locked_before_candidate_aggregation"
    assert contract["lineage"]["active_groups"] == EXPECTED_ACTIVE_GROUPS
    assert contract["lineage"]["active_completions"] == EXPECTED_ACTIVE_COMPLETIONS
    assert contract["lineage"]["same_order_as_original_active_scope"] is True
    boundary = contract["selection_boundary"]
    assert boundary["candidate_replay_records_opened"] is False
    assert boundary["candidate_aggregate_metrics_calculated"] is False
    assert boundary["candidate_rankings_calculated"] is False
    assert boundary["acceptance_gates_applied"] is False
    assert boundary["winner_selected"] is False
    assert contract["inputs"]["candidate_replay_records_identity_from_manifest_only"][
        "opened_by_contract_builder"
    ] is False
    assert BOOTSTRAP_REPLICATES == 10_000
    assert BOOTSTRAP_SEED == 20_260_812


def test_original_baselines_and_numeric_gates_are_exactly_predeclared():
    contract = build_contract(repo_root=".")
    baseline = contract["locked_original_baseline"]
    assert baseline["active_zero_variance_share"] == 0.0
    assert baseline["active_groups_with_at_least_three_levels"] == 76
    assert baseline["active_groups_with_large_tie_at_least_six"] == 954
    assert baseline["full_training_zero_variance_groups"] == 1_571
    assert baseline["full_training_zero_variance_share"] == pytest.approx(
        1_571 / 3_240
    )

    gates = {gate["id"]: gate for gate in contract["universal_acceptance_gates"]}
    assert gates["G2_active_variation"]["threshold"] == 0.0
    assert gates["G3_distinct_levels"]["threshold"] == 0.50
    assert gates["G4_large_ties"]["threshold"] == 0.50
    assert gates["G5_pairwise_resolution"]["point_threshold"] == 0.10
    assert gates["G6_known_utility_alignment"]["ci_lower_threshold"] == -0.02
    assert gates["G7_harmful_coverage"]["ci_upper_threshold"] == 0.05
    assert gates["G8_field_dominance"]["threshold"] == 0.20
    assert gates["G9_cb_class_dominance"]["threshold"] == 0.15
    assert gates["G10_full_training_variation"]["threshold"] == 0.40
    assert gates["G10_full_training_variation"][
        "requires_separate_full_scope_raw_replay"
    ] is True


def test_published_contract_matches_a_deterministic_rebuild_when_present():
    published_path = "runs/grpo-run2-comparison-contract.json"
    try:
        with open(published_path, encoding="utf-8") as handle:
            published = json.load(handle)
    except FileNotFoundError:
        pytest.skip("comparison contract has not been published yet")
    assert published == build_contract(repo_root=".")
