#!/usr/bin/env python3
"""Group-aware analysis core for GRPO Run 2 reward candidates.

This first implementation layer intentionally performs no file I/O. It accepts
already-materialized observations, validates synthetic k=8 groups, calculates
the locked descriptive and ordering metrics, and bootstraps paired product
groups. A later adapter will extract these observations from the hash-locked
replay ledger after this arithmetic is proven.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evalharness.paired_bootstrap import linear_percentile
from training.run2_comparison_contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    EXPECTED_CANDIDATES,
    EXPECTED_GROUP_SIZE,
    canonical_number,
    comparison_sign,
    group_reward_shape,
)


VERSION = "grpo-run2-candidate-analysis-core-v1"
ORIGINAL = "original"
REWARD_LABELS = (ORIGINAL, *EXPECTED_CANDIDATES)
KNOWN_UTILITY = "canonical_known_utility"
KNOWN_COVERAGE = "known_coverage"


@dataclass(frozen=True)
class GroupObservation:
    """One product's aligned rewards and analysis targets for eight rollouts."""

    group_id: str
    rewards: Mapping[str, Sequence[float]]
    targets: Mapping[str, Sequence[float | None]]


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _number_key(value: float) -> str:
    value = canonical_number(value)
    return str(int(value)) if value.is_integer() else f"{value:.12g}"


def numeric_distribution(values: Sequence[float]) -> dict[str, Any]:
    """Describe a nonempty numeric sequence with the contract's fixed views."""
    if not values:
        raise ValueError("distribution requires at least one value")
    numeric = [_finite(value, "distribution value") for value in values]
    ordered = sorted(numeric)
    return {
        "count": len(numeric),
        "minimum": ordered[0],
        "p05": linear_percentile(ordered, 0.05),
        "p25": linear_percentile(ordered, 0.25),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p75": linear_percentile(ordered, 0.75),
        "p95": linear_percentile(ordered, 0.95),
        "maximum": ordered[-1],
        "population_std": statistics.pstdev(ordered),
        "histogram": {
            _number_key(value): count
            for value, count in sorted(Counter(canonical_number(v) for v in numeric).items())
        },
    }


def _validate_group_values(
    values: Sequence[float | None], *, name: str, allow_missing: bool
) -> tuple[float | None, ...]:
    if len(values) != EXPECTED_GROUP_SIZE:
        raise ValueError(f"{name} must contain exactly {EXPECTED_GROUP_SIZE} values")
    checked: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            if not allow_missing:
                raise ValueError(f"{name}[{index}] cannot be missing")
            checked.append(None)
        else:
            checked.append(_finite(value, f"{name}[{index}]"))
    return tuple(checked)


def _validate_observations(
    observations: Sequence[GroupObservation],
) -> tuple[GroupObservation, ...]:
    if not observations:
        raise ValueError("at least one group observation is required")
    seen: set[str] = set()
    normalized: list[GroupObservation] = []
    target_names: tuple[str, ...] | None = None
    for position, observation in enumerate(observations):
        if not isinstance(observation.group_id, str) or not observation.group_id:
            raise ValueError(f"group {position} has an invalid group_id")
        if observation.group_id in seen:
            raise ValueError(f"duplicate group_id: {observation.group_id}")
        seen.add(observation.group_id)
        if tuple(observation.rewards) != REWARD_LABELS:
            raise ValueError(
                f"{observation.group_id}: reward labels/order must be {REWARD_LABELS}"
            )
        current_targets = tuple(observation.targets)
        if target_names is None:
            target_names = current_targets
        elif current_targets != target_names:
            raise ValueError(f"{observation.group_id}: target labels/order drifted")
        if KNOWN_UTILITY not in observation.targets or KNOWN_COVERAGE not in observation.targets:
            raise ValueError(
                f"{observation.group_id}: known utility and coverage targets are required"
            )
        rewards = {
            label: _validate_group_values(
                observation.rewards[label],
                name=f"{observation.group_id}.rewards.{label}",
                allow_missing=False,
            )
            for label in REWARD_LABELS
        }
        targets = {
            label: _validate_group_values(
                values,
                name=f"{observation.group_id}.targets.{label}",
                allow_missing=True,
            )
            for label, values in observation.targets.items()
        }
        normalized.append(
            GroupObservation(
                group_id=observation.group_id,
                rewards=rewards,
                targets=targets,
            )
        )
    return tuple(normalized)


def summarize_reward_groups(
    reward_groups: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Summarize reward shape while keeping each product group explicit."""
    if not reward_groups:
        raise ValueError("at least one reward group is required")
    checked = {
        group_id: tuple(
            value
            for value in _validate_group_values(
                values, name=f"reward group {group_id}", allow_missing=False
            )
            if value is not None
        )
        for group_id, values in reward_groups.items()
    }
    shapes = {group_id: group_reward_shape(values) for group_id, values in checked.items()}
    completions = [value for values in checked.values() for value in values]
    group_means = [statistics.fmean(values) for values in checked.values()]
    group_variances = [
        float(shape["population_variance_own_scale"]) for shape in shapes.values()
    ]
    distinct = [int(shape["unique_reward_levels"]) for shape in shapes.values()]
    largest_ties = [int(shape["largest_tie_size"]) for shape in shapes.values()]
    discrimination = [
        float(shape["pairwise_discrimination_rate"]) for shape in shapes.values()
    ]
    groups = len(checked)
    zero_variance = sum(levels == 1 for levels in distinct)
    at_least_three = sum(levels >= 3 for levels in distinct)
    large_ties = sum(size >= 6 for size in largest_ties)
    return {
        "groups": groups,
        "completions": len(completions),
        "completion_reward_distribution": numeric_distribution(completions),
        "group_mean_distribution": numeric_distribution(group_means),
        "within_group_variance_own_scale_distribution": numeric_distribution(
            group_variances
        ),
        "pairwise_discrimination_distribution": numeric_distribution(discrimination),
        "unique_reward_levels_per_group_histogram": {
            str(value): count for value, count in sorted(Counter(distinct).items())
        },
        "largest_tie_size_per_group_histogram": {
            str(value): count for value, count in sorted(Counter(largest_ties).items())
        },
        "zero_variance_groups": zero_variance,
        "zero_variance_share": zero_variance / groups,
        "groups_with_at_least_three_levels": at_least_three,
        "groups_with_at_least_three_levels_share": at_least_three / groups,
        "groups_with_largest_tie_at_least_six": large_ties,
        "groups_with_largest_tie_at_least_six_share": large_ties / groups,
        "group_values_for_paired_analysis": {
            group_id: {
                "pairwise_discrimination_rate": shapes[group_id][
                    "pairwise_discrimination_rate"
                ]
            }
            for group_id in checked
        },
    }


def _directional_alignment_optional(
    rewards: Sequence[float], targets: Sequence[float | None]
) -> dict[str, int | float | None]:
    concordant = discordant = reward_ties = 0
    for left in range(EXPECTED_GROUP_SIZE):
        for right in range(left + 1, EXPECTED_GROUP_SIZE):
            if targets[left] is None or targets[right] is None:
                continue
            target_direction = comparison_sign(targets[left], targets[right])
            if target_direction == 0:
                continue
            reward_direction = comparison_sign(rewards[left], rewards[right])
            if reward_direction == 0:
                reward_ties += 1
            elif reward_direction == target_direction:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant + reward_ties
    return {
        "comparable_pairs": comparable,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "reward_ties": reward_ties,
        "net_alignment": (
            (concordant - discordant) / comparable if comparable else None
        ),
    }


def summarize_directional_alignment_groups(
    reward_groups: Mapping[str, Sequence[float]],
    target_groups: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    """Summarize group-first directional alignment with optional target cells."""
    if set(reward_groups) != set(target_groups):
        raise ValueError("reward and target group IDs must match exactly")
    if not reward_groups:
        raise ValueError("at least one paired group is required")
    group_results: dict[str, dict[str, int | float | None]] = {}
    for group_id in reward_groups:
        rewards = _validate_group_values(
            reward_groups[group_id], name=f"{group_id}.reward", allow_missing=False
        )
        targets = _validate_group_values(
            target_groups[group_id], name=f"{group_id}.target", allow_missing=True
        )
        group_results[group_id] = _directional_alignment_optional(
            tuple(value for value in rewards if value is not None), targets
        )
    contributing = {
        group_id: result
        for group_id, result in group_results.items()
        if result["net_alignment"] is not None
    }
    if not contributing:
        return {
            "groups_total": len(group_results),
            "groups_contributing": 0,
            "comparable_pairs": 0,
            "concordant_pairs": 0,
            "discordant_pairs": 0,
            "reward_ties": 0,
            "group_net_alignment_distribution": None,
            "group_values_for_paired_analysis": {},
        }
    values = {
        group_id: float(result["net_alignment"])
        for group_id, result in contributing.items()
    }
    return {
        "groups_total": len(group_results),
        "groups_contributing": len(contributing),
        "comparable_pairs": sum(int(result["comparable_pairs"]) for result in contributing.values()),
        "concordant_pairs": sum(int(result["concordant_pairs"]) for result in contributing.values()),
        "discordant_pairs": sum(int(result["discordant_pairs"]) for result in contributing.values()),
        "reward_ties": sum(int(result["reward_ties"]) for result in contributing.values()),
        "group_net_alignment_distribution": numeric_distribution(list(values.values())),
        "group_values_for_paired_analysis": values,
    }


def _harmful_coverage_optional(
    rewards: Sequence[float],
    coverages: Sequence[float | None],
    utilities: Sequence[float | None],
) -> dict[str, int | float | None]:
    harmful = safe = reward_ties = 0
    for left in range(EXPECTED_GROUP_SIZE):
        for right in range(left + 1, EXPECTED_GROUP_SIZE):
            if (
                coverages[left] is None
                or coverages[right] is None
                or utilities[left] is None
                or utilities[right] is None
            ):
                continue
            coverage_direction = comparison_sign(coverages[left], coverages[right])
            utility_direction = comparison_sign(utilities[left], utilities[right])
            if coverage_direction == 0 or utility_direction == 0:
                continue
            if coverage_direction == utility_direction:
                continue
            reward_direction = comparison_sign(rewards[left], rewards[right])
            if reward_direction == 0:
                reward_ties += 1
            elif reward_direction == coverage_direction:
                harmful += 1
            else:
                safe += 1
    comparable = harmful + safe + reward_ties
    return {
        "comparable_pairs": comparable,
        "harmful_preferences": harmful,
        "safe_preferences": safe,
        "reward_ties": reward_ties,
        "harmful_preference_score": (
            (harmful + 0.5 * reward_ties) / comparable if comparable else None
        ),
    }


def summarize_harmful_coverage_groups(
    reward_groups: Mapping[str, Sequence[float]],
    coverage_groups: Mapping[str, Sequence[float | None]],
    utility_groups: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    """Summarize harmful coverage preference with equal product weighting."""
    if not (set(reward_groups) == set(coverage_groups) == set(utility_groups)):
        raise ValueError("reward, coverage and utility group IDs must match exactly")
    group_results: dict[str, dict[str, int | float | None]] = {}
    for group_id in reward_groups:
        rewards = _validate_group_values(
            reward_groups[group_id], name=f"{group_id}.reward", allow_missing=False
        )
        coverages = _validate_group_values(
            coverage_groups[group_id], name=f"{group_id}.coverage", allow_missing=True
        )
        utilities = _validate_group_values(
            utility_groups[group_id], name=f"{group_id}.utility", allow_missing=True
        )
        group_results[group_id] = _harmful_coverage_optional(
            tuple(value for value in rewards if value is not None), coverages, utilities
        )
    contributing = {
        group_id: result
        for group_id, result in group_results.items()
        if result["harmful_preference_score"] is not None
    }
    if not contributing:
        return {
            "groups_total": len(group_results),
            "groups_contributing": 0,
            "comparable_pairs": 0,
            "harmful_preferences": 0,
            "safe_preferences": 0,
            "reward_ties": 0,
            "group_score_distribution": None,
            "group_values_for_paired_analysis": {},
        }
    values = {
        group_id: float(result["harmful_preference_score"])
        for group_id, result in contributing.items()
    }
    return {
        "groups_total": len(group_results),
        "groups_contributing": len(contributing),
        "comparable_pairs": sum(int(result["comparable_pairs"]) for result in contributing.values()),
        "harmful_preferences": sum(int(result["harmful_preferences"]) for result in contributing.values()),
        "safe_preferences": sum(int(result["safe_preferences"]) for result in contributing.values()),
        "reward_ties": sum(int(result["reward_ties"]) for result in contributing.values()),
        "group_score_distribution": numeric_distribution(list(values.values())),
        "group_values_for_paired_analysis": values,
    }


def _bootstrap_summary(values: Sequence[float], confidence: float) -> dict[str, Any]:
    alpha = 1.0 - confidence
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "ci": [
            linear_percentile(values, alpha / 2.0),
            linear_percentile(values, 1.0 - alpha / 2.0),
        ],
    }


def paired_group_bootstrap(
    baseline_by_group: Mapping[str, float],
    candidate_by_group: Mapping[str, float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Bootstrap a candidate-minus-baseline mean using whole paired groups."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    if set(baseline_by_group) != set(candidate_by_group):
        raise ValueError("baseline and candidate group IDs must match exactly")
    group_ids = list(baseline_by_group)
    if not group_ids:
        raise ValueError("paired bootstrap requires at least one group")
    baseline = [_finite(baseline_by_group[group_id], "baseline group value") for group_id in group_ids]
    candidate = [_finite(candidate_by_group[group_id], "candidate group value") for group_id in group_ids]
    baseline_point = statistics.fmean(baseline)
    candidate_point = statistics.fmean(candidate)

    rng = random.Random(seed)
    stream_hash = hashlib.sha256()
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    deltas: list[float] = []
    for replicate in range(replicates):
        indices = [rng.randrange(len(group_ids)) for _ in group_ids]
        baseline_value = statistics.fmean(baseline[index] for index in indices)
        candidate_value = statistics.fmean(candidate[index] for index in indices)
        delta = candidate_value - baseline_value
        baseline_samples.append(baseline_value)
        candidate_samples.append(candidate_value)
        deltas.append(delta)
        stream_hash.update(
            (
                json.dumps(
                    [replicate, baseline_value, candidate_value, delta],
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
    return {
        "method": "paired nonparametric product-group bootstrap, percentile interval",
        "pairing_unit": "entire k=8 product group",
        "completion_level_resampling": False,
        "groups_per_replicate": len(group_ids),
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
        "percentile_method": "linear interpolation at sorted index (n - 1) * p",
        "replicate_stream_sha256": stream_hash.hexdigest(),
        "baseline": {
            "point": baseline_point,
            **_bootstrap_summary(baseline_samples, confidence),
        },
        "candidate": {
            "point": candidate_point,
            **_bootstrap_summary(candidate_samples, confidence),
        },
        "delta_candidate_minus_baseline": {
            "point": candidate_point - baseline_point,
            **_bootstrap_summary(deltas, confidence),
            "fraction_below_zero": sum(value < 0.0 for value in deltas) / replicates,
            "fraction_equal_zero": sum(value == 0.0 for value in deltas) / replicates,
            "fraction_above_zero": sum(value > 0.0 for value in deltas) / replicates,
        },
    }


def _paired_overlap(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    shared = [group_id for group_id in baseline if group_id in candidate]
    return (
        {group_id: baseline[group_id] for group_id in shared},
        {group_id: candidate[group_id] for group_id in shared},
    )


def analyze_group_observations(
    observations: Sequence[GroupObservation],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Run the locked core analysis on already-materialized group observations."""
    observations = _validate_observations(observations)
    reward_groups = {
        label: {observation.group_id: observation.rewards[label] for observation in observations}
        for label in REWARD_LABELS
    }
    target_groups = {
        target: {observation.group_id: observation.targets[target] for observation in observations}
        for target in observations[0].targets
    }
    reward_summaries = {
        label: summarize_reward_groups(groups) for label, groups in reward_groups.items()
    }
    alignments = {
        label: {
            target: summarize_directional_alignment_groups(groups, target_groups[target])
            for target in target_groups
        }
        for label, groups in reward_groups.items()
    }
    coverage = {
        label: summarize_harmful_coverage_groups(
            groups, target_groups[KNOWN_COVERAGE], target_groups[KNOWN_UTILITY]
        )
        for label, groups in reward_groups.items()
    }

    paired: dict[str, Any] = {}
    for candidate in EXPECTED_CANDIDATES:
        candidate_metrics: dict[str, Any] = {}
        baseline_resolution = {
            group_id: values["pairwise_discrimination_rate"]
            for group_id, values in reward_summaries[ORIGINAL][
                "group_values_for_paired_analysis"
            ].items()
        }
        candidate_resolution = {
            group_id: values["pairwise_discrimination_rate"]
            for group_id, values in reward_summaries[candidate][
                "group_values_for_paired_analysis"
            ].items()
        }
        candidate_metrics["pairwise_discrimination"] = paired_group_bootstrap(
            baseline_resolution,
            candidate_resolution,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
            confidence=confidence,
        )
        for metric_name, baseline_values, candidate_values in (
            (
                "canonical_known_utility_net_alignment",
                alignments[ORIGINAL][KNOWN_UTILITY]["group_values_for_paired_analysis"],
                alignments[candidate][KNOWN_UTILITY]["group_values_for_paired_analysis"],
            ),
            (
                "harmful_coverage_preference",
                coverage[ORIGINAL]["group_values_for_paired_analysis"],
                coverage[candidate]["group_values_for_paired_analysis"],
            ),
        ):
            paired_baseline, paired_candidate = _paired_overlap(
                baseline_values, candidate_values
            )
            candidate_metrics[metric_name] = (
                paired_group_bootstrap(
                    paired_baseline,
                    paired_candidate,
                    seed=bootstrap_seed,
                    replicates=bootstrap_replicates,
                    confidence=confidence,
                )
                if paired_baseline
                else None
            )
        paired[candidate] = candidate_metrics

    return {
        "version": VERSION,
        "status": "analysis_core_completed",
        "boundary": {
            "input_kind": "already-materialized in-memory observations",
            "file_io_performed": False,
            "real_candidate_replay_opened": False,
            "acceptance_gates_applied": False,
            "winner_selected": False,
        },
        "groups": len(observations),
        "completions": len(observations) * EXPECTED_GROUP_SIZE,
        "group_order": [observation.group_id for observation in observations],
        "reward_summaries": reward_summaries,
        "directional_alignments": alignments,
        "harmful_coverage": coverage,
        "paired_candidate_minus_original": paired,
    }
