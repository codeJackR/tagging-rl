#!/usr/bin/env python3
"""Executable numeric-scale contract for the GRPO Run 2 reward candidates.

This module deliberately does not parse or score model completions.  It only
defines bounded arithmetic selected from training-set structure and publishes
the mechanical checks that must pass before candidate reward replay begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from labeling.records import LabelStatus
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.replay_original_reward import (
    DEFAULT_DIFFICULTY_MANIFEST,
    DEFAULT_PACK,
    DEFAULT_POOL_DATA,
    DEFAULT_POOL_MANIFEST,
    DEFAULT_ROLLOUTS,
    DEFAULT_SFT_SPLIT,
    load_locked_inputs,
)
from verifier import load_pack


VERSION = "grpo-run2-reward-scale-contract-v1"
PAYOFF_CONTRACT_VERSION = "grpo-run2-reward-payoffs-v1"
DEFAULT_PAYOFF_CONTRACT = "W2_GRPO_RUN2_REWARD_PAYOFFS.md"
DEFAULT_ORIGINAL_REPLAY = "runs/grpo-run2-original-reward-training-replay.json"
DEFAULT_OUTPUT = "runs/grpo-run2-reward-scale-contract.json"

CORRECT = 1.0
ABSTAIN = 0.0
WRONG = -1.0
UNKNOWN_ABSTAIN = 1.0
UNKNOWN_COMMIT = -1.0

# Exact cell counts in the locked 1,438-row active training pool.  The artifact
# builder recomputes and rejects drift instead of silently reusing them.
KNOWN_CELLS = 12_533
UNKNOWN_CELLS = 9_037
TOTAL_CELLS = KNOWN_CELLS + UNKNOWN_CELLS
KNOWN_MIX_WEIGHT = KNOWN_CELLS / TOTAL_CELLS
UNKNOWN_MIX_WEIGHT = UNKNOWN_CELLS / TOTAL_CELLS

CLASS_WEIGHT_MIN = 0.5
CLASS_WEIGHT_MAX = 2.0
RULE_VIOLATION_CAP = 3
RULE_MAXIMUM_COST = 0.15
MALFORMED_FLOOR = -1.25


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def details_payoff(set_f1: float) -> float:
    """Map set-F1 from [0, 1] onto WRONG..CORRECT."""
    set_f1 = _finite(set_f1, "set_f1")
    if not 0.0 <= set_f1 <= 1.0:
        raise ValueError("set_f1 must be between 0 and 1")
    return 2.0 * set_f1 - 1.0


def rule_adjustment(violation_count: int) -> float:
    """Return a non-positive, linear-per-violation cost capped at three."""
    if isinstance(violation_count, bool) or not isinstance(violation_count, int):
        raise TypeError("violation_count must be an integer")
    if violation_count < 0:
        raise ValueError("violation_count cannot be negative")
    return -RULE_MAXIMUM_COST * min(violation_count, RULE_VIOLATION_CAP) / RULE_VIOLATION_CAP


def class_weight(class_support: int, attribute_median_support: float) -> float:
    """Return clipped inverse-square-root support weight for one gold class."""
    if isinstance(class_support, bool) or not isinstance(class_support, int):
        raise TypeError("class_support must be an integer")
    if class_support <= 0:
        raise ValueError("class_support must be positive")
    attribute_median_support = _finite(
        attribute_median_support, "attribute_median_support"
    )
    if attribute_median_support <= 0:
        raise ValueError("attribute_median_support must be positive")
    raw = math.sqrt(attribute_median_support / class_support)
    return min(CLASS_WEIGHT_MAX, max(CLASS_WEIGHT_MIN, raw))


def normalized_mean(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Average bounded field utilities, optionally with positive class weights."""
    if not values:
        raise ValueError("at least one value is required")
    values = [_finite(value, "value") for value in values]
    if any(not WRONG <= value <= CORRECT for value in values):
        raise ValueError("values must lie in the semantic interval [-1, 1]")
    if weights is None:
        weights = [1.0] * len(values)
    if len(weights) != len(values):
        raise ValueError("values and weights must have the same length")
    weights = [_finite(weight, "weight") for weight in weights]
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("weights must be positive")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)


def combine_known_unknown(known_score: float, unknown_score: float | None) -> float:
    """Combine separately normalized components using locked training-cell shares.

    Products with no gold-unknown fields retain their known score instead of
    being compressed by a component that is absent for that product.
    """
    known_score = _finite(known_score, "known_score")
    if not WRONG <= known_score <= CORRECT:
        raise ValueError("known_score must lie in [-1, 1]")
    if unknown_score is None:
        return known_score
    unknown_score = _finite(unknown_score, "unknown_score")
    if not UNKNOWN_COMMIT <= unknown_score <= UNKNOWN_ABSTAIN:
        raise ValueError("unknown_score must lie in [-1, 1]")
    return KNOWN_MIX_WEIGHT * known_score + UNKNOWN_MIX_WEIGHT * unknown_score


def valid_total(semantic_score: float, violation_count: int) -> float:
    """Combine a valid record's semantic score and bounded rule adjustment."""
    semantic_score = _finite(semantic_score, "semantic_score")
    if not WRONG <= semantic_score <= CORRECT:
        raise ValueError("semantic_score must lie in [-1, 1]")
    return semantic_score + rule_adjustment(violation_count)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "minimum": min(numeric),
        "p05": _percentile(numeric, 0.05),
        "p25": _percentile(numeric, 0.25),
        "median": statistics.median(numeric),
        "mean": statistics.fmean(numeric),
        "p75": _percentile(numeric, 0.75),
        "p95": _percentile(numeric, 0.95),
        "maximum": max(numeric),
    }


def _file_metadata(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _field_class_keys(label: Any) -> list[str]:
    if label.status is LabelStatus.NOT_APPLICABLE:
        return ["__not_applicable__"]
    if label.status is LabelStatus.LABELED:
        return list(label.value) if isinstance(label.value, list) else [str(label.value)]
    return []


def _training_structure(inputs: Any, pack: Any) -> dict[str, Any]:
    rows = [inputs.rows_by_sku[sku] for sku in inputs.active_pool_skus]
    known_counts: list[int] = []
    unknown_counts: list[int] = []
    details_set_sizes: list[int] = []
    class_supports: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        if set(row.labels) != set(pack.field_names):
            raise RuntimeError(f"{row.sku_id}: active row does not contain exactly 15 fields")
        known = 0
        unknown = 0
        for field_name in pack.field_names:
            label = row.labels[field_name]
            if label.status is LabelStatus.UNKNOWN:
                unknown += 1
                continue
            known += 1
            keys = _field_class_keys(label)
            class_supports[field_name].update(keys)
            if field_name == "details" and label.status is LabelStatus.LABELED:
                details_set_sizes.append(len(keys))
        known_counts.append(known)
        unknown_counts.append(unknown)

    known_cells = sum(known_counts)
    unknown_cells = sum(unknown_counts)
    if (known_cells, unknown_cells) != (KNOWN_CELLS, UNKNOWN_CELLS):
        raise RuntimeError(
            "locked active-pool cell counts drifted: "
            f"observed={(known_cells, unknown_cells)}, "
            f"expected={(KNOWN_CELLS, UNKNOWN_CELLS)}"
        )

    per_attribute: dict[str, Any] = {}
    all_supports: list[int] = []
    for field_name in pack.field_names:
        support = dict(sorted(class_supports[field_name].items()))
        values = list(support.values())
        all_supports.extend(values)
        median_support = statistics.median(values)
        weights = {
            name: class_weight(count, median_support)
            for name, count in support.items()
        }
        per_attribute[field_name] = {
            "observed_classes": len(support),
            "support": support,
            "support_distribution": _distribution(values),
            "median_positive_support": median_support,
            "derived_weight_minimum": min(weights.values()),
            "derived_weight_maximum": max(weights.values()),
        }

    violation_counts = [
        len(record.rule_violations)
        for sku in inputs.active_pool_skus
        for record in inputs.records_by_sku[sku]
    ]
    return {
        "active_products": len(rows),
        "fields_per_product": len(pack.field_names),
        "known_cells": known_cells,
        "unknown_cells": unknown_cells,
        "known_share_exact": f"{KNOWN_CELLS}/{TOTAL_CELLS}",
        "known_share": KNOWN_MIX_WEIGHT,
        "unknown_share_exact": f"{UNKNOWN_CELLS}/{TOTAL_CELLS}",
        "unknown_share": UNKNOWN_MIX_WEIGHT,
        "known_fields_per_product": _distribution(known_counts),
        "unknown_fields_per_product": _distribution(unknown_counts),
        "details_labeled_set_sizes": {
            "distribution": _distribution(details_set_sizes),
            "histogram": dict(sorted(Counter(details_set_sizes).items())),
        },
        "class_support": {
            "observed_attribute_class_pairs": len(all_supports),
            "distribution": _distribution(all_supports),
            "classes_below_five": sum(value < 5 for value in all_supports),
            "classes_below_ten": sum(value < 10 for value in all_supports),
            "per_attribute": per_attribute,
        },
        "starting_policy_rule_violations": {
            "rollouts": len(violation_counts),
            "distribution": _distribution(violation_counts),
            "histogram": {
                str(key): value for key, value in sorted(Counter(violation_counts).items())
            },
            "rule_inventory": pack.rule_inventory(),
        },
    }


def _proofs() -> dict[str, bool]:
    known_order = CORRECT > ABSTAIN > WRONG > MALFORMED_FLOOR
    details_order = (
        details_payoff(0.0) == WRONG
        and details_payoff(0.5) == ABSTAIN
        and details_payoff(1.0) == CORRECT
        and all(details_payoff(a) < details_payoff(b) for a, b in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)))
    )
    rule_order = all(
        rule_adjustment(v) > rule_adjustment(v + 1)
        for v in range(RULE_VIOLATION_CAP)
    ) and rule_adjustment(RULE_VIOLATION_CAP) == rule_adjustment(RULE_VIOLATION_CAP + 1)
    worst_valid = valid_total(WRONG, RULE_VIOLATION_CAP)
    bounded_valid = worst_valid > MALFORMED_FLOOR and valid_total(CORRECT, 0) == CORRECT

    local_uniform = True
    for fields in range(1, 16):
        baseline = normalized_mean([ABSTAIN] * fields)
        local_uniform &= normalized_mean([CORRECT] + [ABSTAIN] * (fields - 1)) > baseline
        local_uniform &= normalized_mean([WRONG] + [ABSTAIN] * (fields - 1)) < baseline

    local_balanced = True
    weights = [CLASS_WEIGHT_MIN, 1.0, CLASS_WEIGHT_MAX]
    baseline = normalized_mean([ABSTAIN] * 3, weights)
    for index in range(3):
        correct = [ABSTAIN] * 3
        wrong = [ABSTAIN] * 3
        correct[index] = CORRECT
        wrong[index] = WRONG
        local_balanced &= normalized_mean(correct, weights) > baseline
        local_balanced &= normalized_mean(wrong, weights) < baseline

    unknown_order = combine_known_unknown(0.0, UNKNOWN_ABSTAIN) > combine_known_unknown(
        0.0, UNKNOWN_COMMIT
    )
    component_bounds = math.isclose(KNOWN_MIX_WEIGHT + UNKNOWN_MIX_WEIGHT, 1.0)
    results = {
        "correct_abstain_wrong_floor_order": known_order,
        "details_endpoints_crossover_and_monotonicity": details_order,
        "each_rule_violation_lowers_reward_until_cap": rule_order,
        "worst_valid_record_remains_above_malformed_floor": bounded_valid,
        "uniform_correct_for_abstain_up_and_wrong_for_abstain_down": local_uniform,
        "class_balanced_correct_for_abstain_up_and_wrong_for_abstain_down": local_balanced,
        "unknown_abstention_ranks_above_unsupported_commitment": unknown_order,
        "known_and_unknown_mix_weights_sum_to_one": component_bounds,
    }
    if not all(results.values()):
        failed = sorted(name for name, passed in results.items() if not passed)
        raise RuntimeError(f"reward-scale proofs failed: {failed}")
    return results


def build_contract(
    *,
    repo_root: str | Path,
    payoff_contract_path: str | Path = DEFAULT_PAYOFF_CONTRACT,
    original_replay_path: str | Path = DEFAULT_ORIGINAL_REPLAY,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    payoff_path = (repo_root / payoff_contract_path).resolve()
    original_path = (repo_root / original_replay_path).resolve()
    payoff_text = payoff_path.read_text(encoding="utf-8")
    if PAYOFF_CONTRACT_VERSION not in payoff_text:
        raise RuntimeError("unexpected ordinal payoff-contract version")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("version") != "grpo-run2-original-reward-replay-v1":
        raise RuntimeError("unexpected original-reward replay version")

    inputs = load_locked_inputs(repo_root=repo_root)
    pack = load_pack(repo_root / DEFAULT_PACK)
    structure = _training_structure(inputs, pack)
    proofs = _proofs()
    implementation_path = Path(__file__).resolve()
    return {
        "version": VERSION,
        "status": "passed",
        "role": "numeric scale selection before candidate reward implementation or replay",
        "selection_boundary": {
            "allowed": "authoritative SFT-training labels and locked starting-policy rule-count structure",
            "prohibited": [
                "legacy frozen 300",
                "SFT validation",
                "probe 100",
                "future confirmation data",
                "candidate reward replay outcomes",
            ],
            "candidate_completion_rewards_calculated": False,
            "hypothesis_tests_performed": False,
        },
        "inputs": {
            "payoff_contract": _file_metadata(payoff_path),
            "original_reward_replay": _file_metadata(original_path),
            "active_pool": inputs.metadata["run2_pool_dataset"],
            "active_pool_manifest": inputs.metadata["run2_pool_manifest"],
            "locked_rollouts": inputs.metadata["rollouts"],
        },
        "implementation": _file_metadata(implementation_path),
        "numeric_contract": {
            "field_payoffs": {
                "correct": CORRECT,
                "abstain": ABSTAIN,
                "wrong": WRONG,
                "unknown_abstain": UNKNOWN_ABSTAIN,
                "unknown_commit": UNKNOWN_COMMIT,
            },
            "semantic_component_range": [WRONG, CORRECT],
            "known_unknown_combination": {
                "known_weight_exact": f"{KNOWN_CELLS}/{TOTAL_CELLS}",
                "known_weight": KNOWN_MIX_WEIGHT,
                "unknown_weight_exact": f"{UNKNOWN_CELLS}/{TOTAL_CELLS}",
                "unknown_weight": UNKNOWN_MIX_WEIGHT,
                "absent_unknown_component": "renormalize to the known score",
            },
            "details": {
                "quality": "order-insensitive set-F1",
                "formula": "P(q) = 2*q - 1",
                "abstention_crossover_set_f1": 0.5,
            },
            "class_balancing": {
                "formula": "clip(sqrt(attribute_median_positive_support / class_support), 0.5, 2.0)",
                "minimum_weight": CLASS_WEIGHT_MIN,
                "maximum_weight": CLASS_WEIGHT_MAX,
                "multi_label_field_weight": "mean weight of its gold labels",
                "normalization": "divide weighted utility sum by observed field-weight sum",
            },
            "rule_adjustment": {
                "formula": "-0.15 * min(violation_count, 3) / 3",
                "per_violation_cost_until_cap": RULE_MAXIMUM_COST / RULE_VIOLATION_CAP,
                "violation_cap": RULE_VIOLATION_CAP,
                "maximum_cost": RULE_MAXIMUM_COST,
            },
            "malformed_floor": MALFORMED_FLOOR,
            "valid_total_range": [valid_total(WRONG, RULE_VIOLATION_CAP), CORRECT],
            "all_output_total_range": [MALFORMED_FLOOR, CORRECT],
        },
        "training_structure": structure,
        "proofs": proofs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--payoff-contract", default=DEFAULT_PAYOFF_CONTRACT)
    parser.add_argument("--original-replay", default=DEFAULT_ORIGINAL_REPLAY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    contract = build_contract(
        repo_root=repo_root,
        payoff_contract_path=args.payoff_contract,
        original_replay_path=args.original_replay,
    )
    write_exclusive_atomic_json(output, contract)
    print(json.dumps({"output": str(output), "status": contract["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
