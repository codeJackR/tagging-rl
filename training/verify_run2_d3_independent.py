#!/usr/bin/env python3
"""Independently verify the published Run 2 D3 aggregate.

This verifier intentionally imports only Python's standard library. It does
not reuse the production analyzer, replay adapter, result contract, reward
implementation, or project artifact helpers. It streams the locked raw replay,
reconstructs the published measurements, and writes a separate exclusive JSON
verification report. It never applies Gates G1-G9 or selects a candidate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "grpo-run2-d3-independent-verification-v1"
RESULT_PATH = "runs/grpo-run2-d3-candidate-analysis.json"
REPORT_PATH = "runs/grpo-run2-d3-independent-verification.json"
RESULT_BYTES = 16_079_523
RESULT_SHA256 = "5812b23c1aae07cbfda923a7715db76f665bb072f39ec37a20592a41d9cbc7fb"
MANIFEST_PATH = "runs/grpo-run2-candidate-replay-manifest.json"
REPLAY_PATH = "runs/grpo-run2-candidate-replay-records.jsonl.gz"
CONTRACT_PATH = "runs/grpo-run2-comparison-contract.json"
CLASS_WEIGHTS_PATH = "runs/grpo-run2-cb-class-weights.json"
SOURCE_IDENTITIES = {
    "manifest": (6_435, "e10c3c47bb54fe0ad4bd07e68966401be71771d2bf134aa2b29dbb9c1683163e"),
    "records": (1_921_202, "30e3ea8681ca80de5737cc5928b8a755e8b14cd36f6c0a67e00385a8408be38a"),
    "comparison_contract": (12_079, "8692291af2319c33a9a6548c1a6530f8c61da0c04e27e69eaa048584245e1142"),
    "class_weights": (27_446, "7b53323a7f1c170fa68c6b1a0d1356c67fd827f70f466ba2972b857418f4ab37"),
}
EXPECTED_GROUPS = 1_438
GROUP_SIZE = 8
EXPECTED_COMPLETIONS = EXPECTED_GROUPS * GROUP_SIZE
PAIR_COUNT = 28
DECIMALS = 12
SEED = 20_260_812
REPLICATES = 10_000
CONFIDENCE = 0.95
REWARDS = ("original", "U", "UA", "CB")
CANDIDATES = ("U", "UA", "CB")
TARGETS = (
    "canonical_known_utility",
    "known_exact_rate",
    "known_coverage",
    "selective_correctness",
    "unknown_abstention_rate",
    "rule_quality",
    "class_balanced_known_utility",
)
KNOWN_WEIGHT = 12_533 / 21_570
UNKNOWN_WEIGHT = 9_037 / 21_570


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _ordered_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("comparison value must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("comparison value must be finite")
    rounded = round(number, DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _sign(left: float, right: float) -> int:
    left = _canonical(left)
    right = _canonical(right)
    return (left > right) - (left < right)


def _number_key(value: float) -> str:
    value = _canonical(value)
    return str(int(value)) if value.is_integer() else f"{value:.12g}"


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric or any(not math.isfinite(value) for value in numeric):
        raise ValueError("distribution requires finite values")
    ordered = sorted(numeric)
    return {
        "count": len(numeric),
        "minimum": ordered[0],
        "p05": _percentile(ordered, 0.05),
        "p25": _percentile(ordered, 0.25),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p75": _percentile(ordered, 0.75),
        "p95": _percentile(ordered, 0.95),
        "maximum": ordered[-1],
        "population_std": statistics.pstdev(ordered),
        "histogram": {
            _number_key(value): count
            for value, count in sorted(Counter(_canonical(v) for v in numeric).items())
        },
    }


def _shape(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != GROUP_SIZE:
        raise ValueError("reward group must contain eight values")
    canonical = [_canonical(value) for value in values]
    counts = Counter(canonical)
    tied = sum(math.comb(count, 2) for count in counts.values())
    return {
        "unique": len(counts),
        "largest_tie": max(counts.values()),
        "variance": statistics.pvariance(canonical),
        "discrimination": (PAIR_COUNT - tied) / PAIR_COUNT,
    }


def _reward_summary(groups: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    shapes = {group_id: _shape(values) for group_id, values in groups.items()}
    completions = [value for values in groups.values() for value in values]
    means = [statistics.fmean(values) for values in groups.values()]
    distinct = [shape["unique"] for shape in shapes.values()]
    ties = [shape["largest_tie"] for shape in shapes.values()]
    variances = [shape["variance"] for shape in shapes.values()]
    discrimination = [shape["discrimination"] for shape in shapes.values()]
    total = len(groups)
    zero = sum(value == 1 for value in distinct)
    three = sum(value >= 3 for value in distinct)
    large = sum(value >= 6 for value in ties)
    return {
        "groups": total,
        "completions": len(completions),
        "completion_reward_distribution": _distribution(completions),
        "group_mean_distribution": _distribution(means),
        "within_group_variance_own_scale_distribution": _distribution(variances),
        "pairwise_discrimination_distribution": _distribution(discrimination),
        "unique_reward_levels_per_group_histogram": {
            str(value): count for value, count in sorted(Counter(distinct).items())
        },
        "largest_tie_size_per_group_histogram": {
            str(value): count for value, count in sorted(Counter(ties).items())
        },
        "zero_variance_groups": zero,
        "zero_variance_share": zero / total,
        "groups_with_at_least_three_levels": three,
        "groups_with_at_least_three_levels_share": three / total,
        "groups_with_largest_tie_at_least_six": large,
        "groups_with_largest_tie_at_least_six_share": large / total,
        "group_values_for_paired_analysis": {
            group_id: {"pairwise_discrimination_rate": shapes[group_id]["discrimination"]}
            for group_id in groups
        },
    }


def _alignment(rewards: Sequence[float], targets: Sequence[float | None]) -> dict[str, Any]:
    concordant = discordant = ties = 0
    for left in range(GROUP_SIZE):
        for right in range(left + 1, GROUP_SIZE):
            if targets[left] is None or targets[right] is None:
                continue
            target_direction = _sign(targets[left], targets[right])
            if target_direction == 0:
                continue
            reward_direction = _sign(rewards[left], rewards[right])
            if reward_direction == 0:
                ties += 1
            elif reward_direction == target_direction:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant + ties
    return {
        "comparable_pairs": comparable,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "reward_ties": ties,
        "value": (concordant - discordant) / comparable if comparable else None,
    }


def _alignment_summary(
    reward_groups: Mapping[str, Sequence[float]],
    target_groups: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    results = {
        group_id: _alignment(reward_groups[group_id], target_groups[group_id])
        for group_id in reward_groups
    }
    contributing = {key: value for key, value in results.items() if value["value"] is not None}
    if not contributing:
        return {
            "groups_total": len(results), "groups_contributing": 0,
            "comparable_pairs": 0, "concordant_pairs": 0,
            "discordant_pairs": 0, "reward_ties": 0,
            "group_net_alignment_distribution": None,
            "group_values_for_paired_analysis": {},
        }
    values = {key: float(value["value"]) for key, value in contributing.items()}
    return {
        "groups_total": len(results),
        "groups_contributing": len(contributing),
        "comparable_pairs": sum(value["comparable_pairs"] for value in contributing.values()),
        "concordant_pairs": sum(value["concordant_pairs"] for value in contributing.values()),
        "discordant_pairs": sum(value["discordant_pairs"] for value in contributing.values()),
        "reward_ties": sum(value["reward_ties"] for value in contributing.values()),
        "group_net_alignment_distribution": _distribution(list(values.values())),
        "group_values_for_paired_analysis": values,
    }


def _harmful(
    rewards: Sequence[float],
    coverages: Sequence[float | None],
    utilities: Sequence[float | None],
) -> dict[str, Any]:
    harmful = safe = ties = 0
    for left in range(GROUP_SIZE):
        for right in range(left + 1, GROUP_SIZE):
            if any(value is None for value in (
                coverages[left], coverages[right], utilities[left], utilities[right]
            )):
                continue
            coverage_direction = _sign(coverages[left], coverages[right])
            utility_direction = _sign(utilities[left], utilities[right])
            if coverage_direction == 0 or utility_direction == 0:
                continue
            if coverage_direction == utility_direction:
                continue
            reward_direction = _sign(rewards[left], rewards[right])
            if reward_direction == 0:
                ties += 1
            elif reward_direction == coverage_direction:
                harmful += 1
            else:
                safe += 1
    comparable = harmful + safe + ties
    return {
        "comparable_pairs": comparable,
        "harmful_preferences": harmful,
        "safe_preferences": safe,
        "reward_ties": ties,
        "value": (harmful + 0.5 * ties) / comparable if comparable else None,
    }


def _harmful_summary(
    reward_groups: Mapping[str, Sequence[float]],
    coverage_groups: Mapping[str, Sequence[float | None]],
    utility_groups: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    results = {
        group_id: _harmful(
            reward_groups[group_id], coverage_groups[group_id], utility_groups[group_id]
        )
        for group_id in reward_groups
    }
    contributing = {key: value for key, value in results.items() if value["value"] is not None}
    if not contributing:
        return {
            "groups_total": len(results), "groups_contributing": 0,
            "comparable_pairs": 0, "harmful_preferences": 0,
            "safe_preferences": 0, "reward_ties": 0,
            "group_score_distribution": None,
            "group_values_for_paired_analysis": {},
        }
    values = {key: float(value["value"]) for key, value in contributing.items()}
    return {
        "groups_total": len(results),
        "groups_contributing": len(contributing),
        "comparable_pairs": sum(value["comparable_pairs"] for value in contributing.values()),
        "harmful_preferences": sum(value["harmful_preferences"] for value in contributing.values()),
        "safe_preferences": sum(value["safe_preferences"] for value in contributing.values()),
        "reward_ties": sum(value["reward_ties"] for value in contributing.values()),
        "group_score_distribution": _distribution(list(values.values())),
        "group_values_for_paired_analysis": values,
    }


def _bootstrap(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, Any]:
    shared = [group_id for group_id in baseline if group_id in candidate]
    if not shared:
        raise ValueError("bootstrap has no shared groups")
    left = [float(baseline[group_id]) for group_id in shared]
    right = [float(candidate[group_id]) for group_id in shared]
    rng = random.Random(SEED)
    stream = hashlib.sha256()
    left_samples = []
    right_samples = []
    deltas = []
    for replicate in range(REPLICATES):
        indices = [rng.randrange(len(shared)) for _ in shared]
        left_value = statistics.fmean(left[index] for index in indices)
        right_value = statistics.fmean(right[index] for index in indices)
        delta = right_value - left_value
        left_samples.append(left_value)
        right_samples.append(right_value)
        deltas.append(delta)
        stream.update((json.dumps(
            [replicate, left_value, right_value, delta], separators=(",", ":")
        ) + "\n").encode("utf-8"))

    def summary(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        alpha = 1.0 - CONFIDENCE
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "ci": [_percentile(ordered, alpha / 2), _percentile(ordered, 1 - alpha / 2)],
        }

    left_point = statistics.fmean(left)
    right_point = statistics.fmean(right)
    return {
        "method": "paired nonparametric product-group bootstrap, percentile interval",
        "pairing_unit": "entire k=8 product group",
        "completion_level_resampling": False,
        "groups_per_replicate": len(shared),
        "seed": SEED,
        "replicates": REPLICATES,
        "confidence": CONFIDENCE,
        "percentile_method": "linear interpolation at sorted index (n - 1) * p",
        "replicate_stream_sha256": stream.hexdigest(),
        "baseline": {"point": left_point, **summary(left_samples)},
        "candidate": {"point": right_point, **summary(right_samples)},
        "delta_candidate_minus_baseline": {
            "point": right_point - left_point,
            **summary(deltas),
            "fraction_below_zero": sum(value < 0 for value in deltas) / REPLICATES,
            "fraction_equal_zero": sum(value == 0 for value in deltas) / REPLICATES,
            "fraction_above_zero": sum(value > 0 for value in deltas) / REPLICATES,
        },
    }


def _assert_same(observed: Any, expected: Any, name: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{name} object keys differ")
        for key in expected:
            _assert_same(observed[key], expected[key], f"{name}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{name} list shape differs")
        for index, value in enumerate(expected):
            _assert_same(observed[index], value, f"{name}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or isinstance(observed, bool) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{name} numeric value differs: {observed!r} != {expected!r}")
        return
    if observed != expected:
        raise ValueError(f"{name} differs: {observed!r} != {expected!r}")


def _class_supports(value: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for field, attribute in value["weight_map"]["attributes"].items():
        for class_key, entry in attribute["classes"].items():
            support = entry["support"]
            band = (
                "rare_1_4" if support < 5 else "low_5_9" if support < 10
                else "medium_10_49" if support < 50 else "common_50_plus"
            )
            result[(field, class_key)] = {
                "support": support, "weight": entry["weight"], "band": band
            }
    return result


def _stream_replay(path: Path, class_weights: Mapping[str, Any]) -> dict[str, Any]:
    rewards = {label: {} for label in REWARDS}
    targets = {target: {} for target in TARGETS}
    segments = {"product_category": {}, "difficulty_band": {}, "gold_known_field_count": {}}
    field_rows = []
    class_rows = []
    group_order = []
    rollout_keys = []
    supports = _class_supports(class_weights)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            group = json.loads(line)
            if group["group_position"] != position:
                raise ValueError("replay group order drifted")
            group_id = group["sku_id"]
            group_order.append(group_id)
            known = group["gold_known_fields"]
            unknown = group["gold_unknown_fields"]
            completions = group["completions"]
            if len(completions) != GROUP_SIZE:
                raise ValueError("replay completion denominator drifted")
            reward_values = {label: [] for label in REWARDS}
            target_values = {target: [] for target in TARGETS}
            passed = []
            for index, completion in enumerate(completions):
                if completion["rollout_index"] != index:
                    raise ValueError("rollout order drifted")
                source = completion["source_rollout"]
                if source["sku_id"] != group_id or source["rollout_index"] != index:
                    raise ValueError("source rollout identity drifted")
                rollout_keys.append(f"{group_id}\t{index}")
                if not isinstance(source["passed"], bool):
                    raise TypeError("source passed must be boolean")
                passed.append(source["passed"])
                reward_values["original"].append(source_value := completion["original_reward"]["weighted_total"])
                _ = source_value
                candidates = completion["candidates"]
                for candidate in CANDIDATES:
                    reward_values[candidate].append(candidates[candidate]["reward"])
                scorable = source["scorable_labels"]
                correct = source["correct_labels"]
                committed = scorable - len(source["false_abstention_labels"])
                target_values["known_exact_rate"].append(correct / scorable)
                target_values["known_coverage"].append(committed / scorable)
                target_values["selective_correctness"].append(
                    correct / committed if committed else None
                )
                target_values["unknown_abstention_rate"].append(
                    len(source["correct_abstention_labels"]) / unknown if unknown else None
                )
                target_values["rule_quality"].append(-float(len(source["rule_violations"])))
                eligible = candidates["U"]["eligible"]
                target_values["canonical_known_utility"].append(
                    candidates["U"]["known_semantics"]["semantic_score"] if eligible else None
                )
                target_values["class_balanced_known_utility"].append(
                    candidates["CB"]["unknown_aware_semantics"]["known_semantics"]["semantic_score"]
                    if eligible else None
                )
                if not eligible:
                    continue
                known_multiplier = KNOWN_WEIGHT if unknown else 1.0
                unknown_multiplier = UNKNOWN_WEIGHT if unknown else 0.0
                ledgers = {
                    "U": candidates["U"]["known_semantics"],
                    "UA": candidates["UA"]["unknown_aware_semantics"],
                    "CB": candidates["CB"]["unknown_aware_semantics"],
                }
                for outcome in ledgers["U"]["field_outcomes"]:
                    absolute = abs(outcome["utility"] / known)
                    field_rows.append((group_id, "U", outcome["field_name"], "known", absolute))
                for outcome in ledgers["UA"]["known_semantics"]["field_outcomes"]:
                    absolute = abs(outcome["utility"] * known_multiplier / known)
                    field_rows.append((group_id, "UA", outcome["field_name"], "known", absolute))
                if unknown:
                    for outcome in ledgers["UA"]["unknown_semantics"]["field_outcomes"]:
                        absolute = abs(outcome["utility"] * unknown_multiplier / unknown)
                        field_rows.append((group_id, "UA", outcome["field_name"], "unknown", absolute))
                cb_known = ledgers["CB"]["known_semantics"]
                total_weight = cb_known["total_field_weight"]
                for outcome in cb_known["field_outcomes"]:
                    absolute = abs(
                        outcome["utility"] * known_multiplier * outcome["field_weight"] / total_weight
                    )
                    field_rows.append((group_id, "CB", outcome["field_name"], "known", absolute))
                    weight_sum = sum(outcome["gold_class_weights"])
                    for class_key, weight in zip(
                        outcome["gold_class_keys"], outcome["gold_class_weights"], strict=True
                    ):
                        metadata = supports[(outcome["field_name"], class_key)]
                        if not math.isclose(weight, metadata["weight"], rel_tol=0, abs_tol=1e-12):
                            raise ValueError("class weight drifted")
                        class_rows.append((
                            group_id, outcome["field_name"], class_key,
                            metadata["support"], metadata["band"], weight,
                            absolute * weight / weight_sum,
                        ))
                if unknown:
                    for outcome in ledgers["CB"]["unknown_semantics"]["field_outcomes"]:
                        absolute = abs(outcome["utility"] * unknown_multiplier / unknown)
                        field_rows.append((group_id, "CB", outcome["field_name"], "unknown", absolute))

            for label in REWARDS:
                rewards[label][group_id] = reward_values[label]
            for target in TARGETS:
                targets[target][group_id] = target_values[target]
            category = group["gold_record"]["garment_category"]
            segments["product_category"][group_id] = (
                "__not_applicable__" if category is None else
                "__gold_unknown__" if category == "unknown" else category
            )
            pass_rate = sum(passed) / GROUP_SIZE
            segments["difficulty_band"][group_id] = (
                "always_failed" if pass_rate == 0 else
                "low_mixed_1_2_of_8" if pass_rate <= 0.25 else
                "middle_mixed_3_5_of_8" if pass_rate < 0.75 else
                "high_mixed_6_7_of_8" if pass_rate < 1 else "always_passed"
            )
            segments["gold_known_field_count"][group_id] = str(known)
    return {
        "rewards": rewards, "targets": targets, "segments": segments,
        "field_rows": field_rows, "class_rows": class_rows,
        "group_order": group_order, "rollout_keys": rollout_keys,
    }


def _dominance(replay: Mapping[str, Any]) -> dict[str, Any]:
    field_summary = {}
    for candidate in CANDIDATES:
        rows = [row for row in replay["field_rows"] if row[1] == candidate]
        by_field = defaultdict(float)
        counts = Counter()
        by_component = defaultdict(float)
        by_group = defaultdict(lambda: defaultdict(float))
        for group, _candidate, field, component, absolute in rows:
            by_field[field] += absolute
            counts[field] += 1
            by_component[component] += absolute
            by_group[group][field] += absolute
        total = sum(by_field.values())
        fields = {
            field: {
                "absolute_contribution": by_field[field],
                "share": by_field[field] / total if total else 0.0,
                "child_rows": counts[field],
            }
            for field in sorted(by_field)
        }
        largest = min(fields, key=lambda field: (-fields[field]["share"], field))
        per_group = [max(values.values()) / sum(values.values()) for values in by_group.values() if sum(values.values())]
        field_summary[candidate] = {
            "child_rows": len(rows),
            "absolute_contribution_total": total,
            "absolute_contribution_by_component": dict(sorted(by_component.items())),
            "fields": fields,
            "largest_field": largest,
            "largest_field_share": fields[largest]["share"],
            "per_group_largest_field_share_distribution": _distribution(per_group) if per_group else None,
            "groups_with_nonzero_absolute_contribution": len(per_group),
            "dominance_threshold_reference": 0.20,
            "dominance_gate_applied": False,
        }

    by_class = defaultdict(float)
    counts = Counter()
    by_band = defaultdict(float)
    band_counts = Counter()
    by_group = defaultdict(lambda: defaultdict(float))
    metadata = {}
    for group, field, class_key, support, band, weight, absolute in replay["class_rows"]:
        class_id = f"{field}::{class_key}"
        by_class[class_id] += absolute
        counts[class_id] += 1
        by_band[band] += absolute
        band_counts[band] += 1
        by_group[group][class_id] += absolute
        metadata[class_id] = {
            "field_name": field, "class_key": class_key, "class_support": support,
            "class_support_band": band, "class_weight": weight,
        }
    total = sum(by_class.values())
    classes = {
        class_id: {
            "absolute_contribution": by_class[class_id],
            "share": by_class[class_id] / total if total else 0.0,
            "child_rows": counts[class_id],
            **metadata[class_id],
        }
        for class_id in sorted(by_class)
    }
    largest = min(classes, key=lambda key: (-classes[key]["share"], key))
    per_group = [max(values.values()) / sum(values.values()) for values in by_group.values() if sum(values.values())]
    return {
        "version": "grpo-run2-contribution-segment-summaries-v1",
        "groups": EXPECTED_GROUPS,
        "field_contributions": field_summary,
        "cb_class_contributions": {
            "child_rows": len(replay["class_rows"]),
            "absolute_contribution_total": total,
            "classes": classes,
            "support_bands": {
                band: {
                    "absolute_contribution": by_band[band],
                    "share": by_band[band] / total if total else 0.0,
                    "child_rows": band_counts[band],
                }
                for band in sorted(by_band)
            },
            "largest_class": largest,
            "largest_class_share": classes[largest]["share"],
            "per_group_largest_class_share_distribution": _distribution(per_group) if per_group else None,
            "groups_with_nonzero_absolute_contribution": len(per_group),
            "dominance_threshold_reference": 0.15,
            "dominance_gate_applied": False,
        },
        "grain_guardrails": {
            "product_counts_derived_from_group_observations_only": True,
            "field_child_rows_used_as_product_denominator": False,
            "class_child_rows_used_as_product_denominator": False,
            "class_allocations_reconstruct_cb_known_fields": True,
        },
    }


def _summaries(replay: Mapping[str, Any], group_ids: Sequence[str]) -> dict[str, Any]:
    rewards = {
        label: {group_id: replay["rewards"][label][group_id] for group_id in group_ids}
        for label in REWARDS
    }
    targets = {
        target: {group_id: replay["targets"][target][group_id] for group_id in group_ids}
        for target in TARGETS
    }
    reward_summaries = {label: _reward_summary(values) for label, values in rewards.items()}
    alignments = {
        label: {
            target: _alignment_summary(values, targets[target]) for target in TARGETS
        }
        for label, values in rewards.items()
    }
    harmful = {
        label: _harmful_summary(
            values, targets["known_coverage"], targets["canonical_known_utility"]
        )
        for label, values in rewards.items()
    }
    return {"reward_summaries": reward_summaries, "directional_alignments": alignments, "harmful_coverage": harmful}


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify(*, repo_root: str | Path, output_path: str | Path = REPORT_PATH) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    result_path = root / RESULT_PATH
    output = Path(output_path)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if output != (root / REPORT_PATH).resolve():
        raise ValueError(f"verification output must be exactly {REPORT_PATH}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    result_identity = _identity(result_path, root)
    if (result_identity["bytes"], result_identity["sha256"]) != (RESULT_BYTES, RESULT_SHA256):
        raise ValueError("published D3 result identity drifted")
    source_paths = {
        "manifest": root / MANIFEST_PATH,
        "records": root / REPLAY_PATH,
        "comparison_contract": root / CONTRACT_PATH,
        "class_weights": root / CLASS_WEIGHTS_PATH,
    }
    source_identities = {name: _identity(path, root) for name, path in source_paths.items()}
    for name, (size, digest) in SOURCE_IDENTITIES.items():
        if (source_identities[name]["bytes"], source_identities[name]["sha256"]) != (size, digest):
            raise ValueError(f"{name} identity drifted")
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    class_weights = json.loads(source_paths["class_weights"].read_text(encoding="utf-8"))
    replay = _stream_replay(source_paths["records"], class_weights)
    group_order = replay["group_order"]
    if len(group_order) != EXPECTED_GROUPS or len(set(group_order)) != EXPECTED_GROUPS:
        raise ValueError("independent group denominator drifted")
    if len(replay["rollout_keys"]) != EXPECTED_COMPLETIONS:
        raise ValueError("independent completion denominator drifted")
    _assert_same(artifact["analysis_core"]["group_order"], group_order, "group_order")
    if artifact["lineage"]["ordered_sku_sha256"] != _ordered_hash(group_order):
        raise ValueError("ordered SKU hash differs")
    if artifact["lineage"]["ordered_rollout_key_sha256"] != _ordered_hash(replay["rollout_keys"]):
        raise ValueError("ordered rollout hash differs")

    core = _summaries(replay, group_order)
    for key in ("reward_summaries", "directional_alignments", "harmful_coverage"):
        _assert_same(artifact["analysis_core"][key], core[key], f"analysis_core.{key}")

    paired = {}
    original_resolution = {
        key: value["pairwise_discrimination_rate"]
        for key, value in core["reward_summaries"]["original"]["group_values_for_paired_analysis"].items()
    }
    for candidate in CANDIDATES:
        candidate_resolution = {
            key: value["pairwise_discrimination_rate"]
            for key, value in core["reward_summaries"][candidate]["group_values_for_paired_analysis"].items()
        }
        paired[candidate] = {
            "pairwise_discrimination": _bootstrap(original_resolution, candidate_resolution),
            "canonical_known_utility_net_alignment": _bootstrap(
                core["directional_alignments"]["original"]["canonical_known_utility"]["group_values_for_paired_analysis"],
                core["directional_alignments"][candidate]["canonical_known_utility"]["group_values_for_paired_analysis"],
            ),
            "harmful_coverage_preference": _bootstrap(
                core["harmful_coverage"]["original"]["group_values_for_paired_analysis"],
                core["harmful_coverage"][candidate]["group_values_for_paired_analysis"],
            ),
        }
    _assert_same(artifact["analysis_core"]["paired_candidate_minus_original"], paired, "paired")

    dimensions = artifact["contribution_and_segment_diagnostics"]["product_segments"]["dimensions"]
    for dimension, group_values in replay["segments"].items():
        expected_members = defaultdict(list)
        for group_id in group_order:
            expected_members[group_values[group_id]].append(group_id)
        observed_dimension = dimensions[dimension]
        if observed_dimension["group_memberships"] != EXPECTED_GROUPS:
            raise ValueError(f"{dimension} membership denominator drifted")
        if set(observed_dimension["segments"]) != set(expected_members):
            raise ValueError(f"{dimension} segment set drifted")
        for segment, members in expected_members.items():
            observed = observed_dimension["segments"][segment]
            _assert_same(observed["ordered_group_ids"], members, f"{dimension}.{segment}.members")
            segment_core = _summaries(replay, members)
            for key in ("reward_summaries", "directional_alignments", "harmful_coverage"):
                _assert_same(observed[key], segment_core[key], f"{dimension}.{segment}.{key}")

    dominance = _dominance(replay)
    _assert_same(
        artifact["contribution_and_segment_diagnostics"]["dominance"],
        dominance,
        "dominance",
    )
    boundary = artifact["selection_boundary"]
    expected_boundary = {
        "candidate_aggregate_metrics_calculated": True,
        "real_candidate_replay_used": True,
        "acceptance_gates_applied": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
    }
    _assert_same(boundary, expected_boundary, "selection_boundary")
    findings = {
        label: {
            "zero_variance_groups": core["reward_summaries"][label]["zero_variance_groups"],
            "groups_with_at_least_three_levels": core["reward_summaries"][label]["groups_with_at_least_three_levels"],
            "groups_with_largest_tie_at_least_six": core["reward_summaries"][label]["groups_with_largest_tie_at_least_six"],
            "mean_pairwise_discrimination": core["reward_summaries"][label]["pairwise_discrimination_distribution"]["mean"],
        }
        for label in REWARDS
    }
    report = {
        "version": VERSION,
        "status": "independent_d3_verification_passed",
        "method": "Python standard library only; no production analysis imports",
        "inputs": {"result": result_identity, **source_identities},
        "lineage": {
            "groups": len(group_order),
            "completions": len(replay["rollout_keys"]),
            "ordered_sku_sha256": _ordered_hash(group_order),
            "ordered_rollout_key_sha256": _ordered_hash(replay["rollout_keys"]),
        },
        "verified": {
            "source_identities": True,
            "group_and_completion_denominators": True,
            "reward_shapes_and_distributions": True,
            "all_directional_alignments": True,
            "harmful_coverage_preferences": True,
            "all_paired_bootstraps": True,
            "all_segment_memberships_and_summaries": True,
            "field_and_class_contribution_dominance": True,
            "no_selection_boundary": True,
        },
        "findings": findings,
        "selection_boundary": {
            "gates_g1_through_g9_applied": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
            "gpu_training_authorized": False,
        },
    }
    _publish_exclusive(output, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default=REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify(repo_root=args.repo_root, output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "groups": report["lineage"]["groups"],
        "completions": report["lineage"]["completions"],
        "output": REPORT_PATH,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
