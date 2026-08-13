from __future__ import annotations

import pytest

from training.analyze_run2_candidates import (
    GroupObservation,
    analyze_group_observations,
    numeric_distribution,
    paired_group_bootstrap,
    summarize_directional_alignment_groups,
    summarize_harmful_coverage_groups,
    summarize_reward_groups,
)


def _observation(group_id: str, offset: float = 0.0) -> GroupObservation:
    target = tuple(float(value) for value in range(8))
    coverage = tuple(float(8 - value) / 8 for value in range(8))
    return GroupObservation(
        group_id=group_id,
        rewards={
            "original": (0, 0, 0, 0, 1, 1, 1, 1),
            "U": tuple(offset + value / 10 for value in range(8)),
            "UA": tuple(offset + value / 9 for value in range(8)),
            "CB": tuple(offset + value / 8 for value in range(8)),
        },
        targets={
            "canonical_known_utility": target,
            "known_coverage": coverage,
            "known_exact_rate": target,
            "unknown_abstention_rate": tuple(reversed(target)),
            "rule_quality": target,
            "class_balanced_known_utility": target,
        },
    )


def test_numeric_distribution_reports_mean_median_spread_and_quantiles():
    result = numeric_distribution([0, 1, 2, 100])
    assert result["count"] == 4
    assert result["mean"] == 25.75
    assert result["median"] == 1.5
    assert result["p25"] == 0.75
    assert result["p75"] == 26.5
    assert result["minimum"] == 0
    assert result["maximum"] == 100
    assert result["population_std"] > 40


def test_reward_summary_keeps_product_groups_as_the_analysis_unit():
    groups = {
        "a": [0] * 8,
        "b": list(range(8)),
    }
    result = summarize_reward_groups(groups)

    assert result["groups"] == 2
    assert result["completions"] == 16
    assert result["zero_variance_groups"] == 1
    assert result["zero_variance_share"] == 0.5
    assert result["groups_with_at_least_three_levels"] == 1
    assert result["groups_with_at_least_three_levels_share"] == 0.5
    assert result["groups_with_largest_tie_at_least_six"] == 1
    # Group-first mean: (0 discrimination + 1 discrimination) / 2.
    assert result["pairwise_discrimination_distribution"]["mean"] == 0.5


def test_alignment_skips_missing_targets_without_turning_reward_ties_into_wins():
    rewards = {"a": [0] * 8, "b": list(range(8))}
    targets = {
        "a": [None, None, 2, 3, 4, 5, 6, 7],
        "b": list(range(8)),
    }
    result = summarize_directional_alignment_groups(rewards, targets)

    assert result["groups_total"] == 2
    assert result["groups_contributing"] == 2
    assert result["comparable_pairs"] == 15 + 28
    assert result["reward_ties"] == 15
    assert result["group_net_alignment_distribution"]["mean"] == 0.5


def test_harmful_coverage_is_averaged_per_group_not_pooled_by_pair_count():
    rewards = {
        "few_pairs": [0, 1, 0, 0, 0, 0, 0, 0],
        "many_pairs": list(reversed(range(8))),
    }
    coverage = {
        "few_pairs": [0, 1, None, None, None, None, None, None],
        "many_pairs": list(range(8)),
    }
    utility = {
        "few_pairs": [1, 0, None, None, None, None, None, None],
        "many_pairs": list(reversed(range(8))),
    }
    result = summarize_harmful_coverage_groups(rewards, coverage, utility)

    assert result["comparable_pairs"] == 1 + 28
    assert result["groups_contributing"] == 2
    # First group is fully harmful; second is fully safe. Equal group weighting.
    assert result["group_score_distribution"]["mean"] == 0.5


def test_paired_group_bootstrap_is_deterministic_and_preserves_pairing():
    baseline = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    candidate = {group_id: value + 0.25 for group_id, value in baseline.items()}
    first = paired_group_bootstrap(
        baseline, candidate, seed=17, replicates=250
    )
    second = paired_group_bootstrap(
        baseline, candidate, seed=17, replicates=250
    )

    assert first == second
    assert first["completion_level_resampling"] is False
    assert first["groups_per_replicate"] == 4
    delta = first["delta_candidate_minus_baseline"]
    assert delta["point"] == pytest.approx(0.25)
    assert delta["ci"] == pytest.approx([0.25, 0.25])
    assert delta["fraction_above_zero"] == 1.0
    assert len(first["replicate_stream_sha256"]) == 64


def test_integrated_core_analyzes_only_synthetic_in_memory_groups():
    result = analyze_group_observations(
        [_observation("a"), _observation("b", 0.01)],
        bootstrap_seed=23,
        bootstrap_replicates=100,
    )

    assert result["status"] == "analysis_core_completed"
    assert result["groups"] == 2
    assert result["completions"] == 16
    assert result["group_order"] == ["a", "b"]
    assert set(result["reward_summaries"]) == {"original", "U", "UA", "CB"}
    assert result["reward_summaries"]["original"][
        "unique_reward_levels_per_group_histogram"
    ] == {"2": 2}
    assert result["paired_candidate_minus_original"]["U"][
        "pairwise_discrimination"
    ]["delta_candidate_minus_baseline"]["point"] > 0
    assert result["boundary"] == {
        "input_kind": "already-materialized in-memory observations",
        "file_io_performed": False,
        "real_candidate_replay_opened": False,
        "acceptance_gates_applied": False,
        "winner_selected": False,
    }


def test_core_fails_closed_on_misalignment_and_invalid_numbers():
    good = _observation("a")
    with pytest.raises(ValueError, match="duplicate group_id"):
        analyze_group_observations([good, good], bootstrap_replicates=2)

    wrong_order = GroupObservation(
        group_id="bad-order",
        rewards={
            "U": [0] * 8,
            "original": [0] * 8,
            "UA": [0] * 8,
            "CB": [0] * 8,
        },
        targets=good.targets,
    )
    with pytest.raises(ValueError, match="reward labels/order"):
        analyze_group_observations([wrong_order], bootstrap_replicates=2)

    with pytest.raises(ValueError, match="exactly 8"):
        summarize_reward_groups({"short": [0] * 7})
    with pytest.raises(ValueError, match="finite"):
        numeric_distribution([float("nan")])
    with pytest.raises(ValueError, match="group IDs must match"):
        paired_group_bootstrap({"a": 0.0}, {"b": 1.0}, replicates=2)
