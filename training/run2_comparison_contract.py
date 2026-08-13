#!/usr/bin/env python3
"""Lock GRPO Run 2 comparison metrics and acceptance gates before aggregation.

This module may read the raw replay *manifest* to pin its identity, but it never
opens the candidate replay records.  The small metric helpers are executable
definitions for the later D3 aggregator and are tested only on synthetic data
at this stage.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json


VERSION = "grpo-run2-comparison-acceptance-contract-v1"
DEFAULT_DOCUMENT = "W2_GRPO_RUN2_COMPARISON_CONTRACT.md"
DEFAULT_PAYOFF_CONTRACT = "W2_GRPO_RUN2_REWARD_PAYOFFS.md"
DEFAULT_SCALE_CONTRACT = "W2_GRPO_RUN2_REWARD_SCALES.md"
DEFAULT_ORIGINAL_REPLAY = "runs/grpo-run2-original-reward-training-replay.json"
DEFAULT_CANDIDATE_MANIFEST = "runs/grpo-run2-candidate-replay-manifest.json"
DEFAULT_OUTPUT = "runs/grpo-run2-comparison-contract.json"

EXPECTED_ORIGINAL_VERSION = "grpo-run2-original-reward-replay-v1"
EXPECTED_CANDIDATE_REPLAY_VERSION = "grpo-run2-candidate-replay-records-v1"
EXPECTED_CANDIDATES = ("U", "UA", "CB")
EXPECTED_ACTIVE_GROUPS = 1_438
EXPECTED_GROUP_SIZE = 8
EXPECTED_ACTIVE_COMPLETIONS = EXPECTED_ACTIVE_GROUPS * EXPECTED_GROUP_SIZE
PAIR_COUNT_PER_GROUP = math.comb(EXPECTED_GROUP_SIZE, 2)

# Rewards are deterministic decimals on a roughly [-1.25, 4] scale.  Twelve
# decimal places preserve meaningful distinctions while preventing irrelevant
# binary-float dust from creating fake reward levels.
COMPARISON_DECIMALS = 12
BOOTSTRAP_SEED = 20_260_812
BOOTSTRAP_REPLICATES = 10_000
CONFIDENCE_LEVEL = 0.95
MIN_PRIMARY_GROUP_SUPPORT = 200
MIN_SEGMENT_GROUP_SUPPORT = 30


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def canonical_number(value: float) -> float:
    """Return the comparison value used for all ties and directions."""
    rounded = round(_finite(value, "comparison value"), COMPARISON_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def comparison_sign(left: float, right: float) -> int:
    """Compare two numbers after the contract's fixed canonicalization."""
    left = canonical_number(left)
    right = canonical_number(right)
    return (left > right) - (left < right)


def group_reward_shape(rewards: Sequence[float]) -> dict[str, int | float]:
    """Calculate scale-safe tie and resolution metrics for one k=8 group."""
    if len(rewards) != EXPECTED_GROUP_SIZE:
        raise ValueError(f"reward group must contain exactly {EXPECTED_GROUP_SIZE} values")
    values = [canonical_number(value) for value in rewards]
    counts = Counter(values)
    tied_pairs = sum(math.comb(count, 2) for count in counts.values())
    return {
        "unique_reward_levels": len(counts),
        "largest_tie_size": max(counts.values()),
        "tied_pairs": tied_pairs,
        "discriminated_pairs": PAIR_COUNT_PER_GROUP - tied_pairs,
        "pairwise_discrimination_rate": (
            PAIR_COUNT_PER_GROUP - tied_pairs
        ) / PAIR_COUNT_PER_GROUP,
        # Retained for within-candidate description only.  It must never be used
        # to compare the old [0,4] reward directly with dense [-1.25,1] rewards.
        "population_variance_own_scale": statistics.pvariance(values),
    }


def directional_net_alignment(
    rewards: Sequence[float], target: Sequence[float]
) -> dict[str, int | float | None]:
    """Measure whether reward orders target-different pairs correctly.

    Target ties are excluded. Reward ties remain in the denominator and score
    zero, so a coarse reward cannot look accurate merely by refusing to rank.
    """
    if len(rewards) != EXPECTED_GROUP_SIZE or len(target) != EXPECTED_GROUP_SIZE:
        raise ValueError("reward and target groups must both contain exactly 8 values")
    concordant = discordant = reward_ties = 0
    for left, right in itertools.combinations(range(EXPECTED_GROUP_SIZE), 2):
        target_direction = comparison_sign(target[left], target[right])
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


def harmful_coverage_preference(
    rewards: Sequence[float],
    known_coverage: Sequence[float],
    known_utility: Sequence[float],
) -> dict[str, int | float | None]:
    """Score pairs where extra coverage comes with worse known-field utility.

    Preferring the higher-coverage/lower-utility completion scores 1, a reward
    tie scores 0.5, and preferring the lower-coverage/higher-utility completion
    scores 0. Lower is therefore safer, without giving coarse ties free credit.
    """
    if not (
        len(rewards)
        == len(known_coverage)
        == len(known_utility)
        == EXPECTED_GROUP_SIZE
    ):
        raise ValueError("reward, coverage and utility groups must contain exactly 8 values")
    harmful = safe = reward_ties = 0
    for left, right in itertools.combinations(range(EXPECTED_GROUP_SIZE), 2):
        coverage_direction = comparison_sign(
            known_coverage[left], known_coverage[right]
        )
        utility_direction = comparison_sign(
            known_utility[left], known_utility[right]
        )
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


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _file_metadata(path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": _portable(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _share_at_least(histogram: Mapping[str, int], threshold: int, total: int) -> float:
    return sum(count for key, count in histogram.items() if int(key) >= threshold) / total


def _baseline(original: Mapping[str, Any]) -> dict[str, Any]:
    active = original["scopes"]["run2_active_pool"]
    full = original["scopes"]["authoritative_sft_train"]
    active_total = active["channels"]["weighted_total"]
    full_total = full["channels"]["weighted_total"]
    groups = active["groups"]
    if groups != EXPECTED_ACTIVE_GROUPS or active["completions"] != EXPECTED_ACTIVE_COMPLETIONS:
        raise RuntimeError("original active-scope counts drifted")
    distinct_histogram = active_total["unique_reward_values_per_group_histogram"]
    largest_tie_histogram = active_total["largest_tie_size_per_group_histogram"]
    return {
        "active_groups": groups,
        "active_completions": active["completions"],
        "active_zero_variance_share": active_total["zero_variance_share"],
        "active_groups_with_at_least_three_levels": sum(
            count for key, count in distinct_histogram.items() if int(key) >= 3
        ),
        "active_groups_with_at_least_three_levels_share": _share_at_least(
            distinct_histogram, 3, groups
        ),
        "active_groups_with_large_tie_at_least_six": sum(
            count for key, count in largest_tie_histogram.items() if int(key) >= 6
        ),
        "active_groups_with_large_tie_at_least_six_share": _share_at_least(
            largest_tie_histogram, 6, groups
        ),
        "active_unique_level_histogram": distinct_histogram,
        "active_largest_tie_histogram": largest_tie_histogram,
        "full_training_groups": full["groups"],
        "full_training_zero_variance_groups": full_total["zero_variance_groups"],
        "full_training_zero_variance_share": full_total["zero_variance_share"],
    }


def _metric_contract() -> dict[str, Any]:
    return {
        "unit_of_analysis": "one product/SKU rollout group",
        "group_size": EXPECTED_GROUP_SIZE,
        "unordered_pairs_per_group": PAIR_COUNT_PER_GROUP,
        "candidate_order": list(EXPECTED_CANDIDATES),
        "canonicalization": {
            "decimal_places": COMPARISON_DECIMALS,
            "method": "round each finite value to 12 decimal places; canonicalize negative zero",
            "applies_to": "reward ties, target ties and pairwise directions",
        },
        "descriptive_reporting": {
            "completion_reward": [
                "count", "minimum", "p05", "p25", "median", "mean",
                "p75", "p95", "maximum", "population_std",
            ],
            "group_reward": [
                "mean", "median", "population_variance_own_scale",
                "unique_reward_levels", "largest_tie_size",
                "pairwise_discrimination_rate",
            ],
            "warning": (
                "raw variance is descriptive within a reward only; do not compare "
                "its magnitude across the original [0,4] and dense [-1.25,1] scales"
            ),
        },
        "primary_group_metrics": {
            "pairwise_discrimination_rate": (
                "non-tied unordered reward pairs divided by 28; average product groups equally"
            ),
            "directional_net_alignment": (
                "(concordant - discordant) / target-different pairs; reward ties stay in denominator"
            ),
            "harmful_coverage_preference_score": (
                "on higher-coverage/lower-known-utility pairs: harmful=1, tie=0.5, safe=0"
            ),
        },
        "targets": {
            "canonical_known_utility": (
                "Candidate U's per-record normalized known-field utility before rule adjustment; "
                "the shared correct/partial/abstain/wrong policy target"
            ),
            "known_exact_rate": "source_rollout.correct_labels / source_rollout.scorable_labels",
            "known_coverage": (
                "(scorable_labels - count(false_abstention_labels)) / scorable_labels"
            ),
            "selective_correctness": (
                "correct_labels / (scorable_labels - count(false_abstention_labels)); null if denominator zero"
            ),
            "unknown_abstention_rate": (
                "count(correct_abstention_labels) / count(excluded_gold_unknown_labels); null if no unknown gold"
            ),
            "rule_quality": "negative count(rule_violations), so fewer violations is higher",
            "class_balanced_known_utility": (
                "Candidate CB's weighted known semantic score before unknown/rule composition"
            ),
        },
        "group_aggregation": (
            "calculate each rate/alignment inside each eligible group, then report mean and median "
            "across groups; also report contributing group and pair counts"
        ),
        "segments": {
            "required": [
                "product category", "difficulty pass-rate band", "gold-known field count",
                "attribute", "gold class support band",
            ],
            "minimum_groups_for_interpretation": MIN_SEGMENT_GROUP_SUPPORT,
            "smaller_segments": "report support only; do not make directional claims",
        },
        "contribution_accounting": {
            "field_share": (
                "sum absolute post-normalization semantic contribution by field, divided by the "
                "sum across fields; exclude malformed floors and rule adjustment"
            ),
            "class_share": (
                "for CB, allocate a field's absolute contribution across its gold class keys in "
                "proportion to their locked class weights; details unknown is excluded"
            ),
            "rule_share_reported_separately": True,
        },
        "sensitivity": {
            "required": [
                "CB replay with class weights replaced by 1.0",
                "CB replay with cap changed from [0.5,2.0] to [0.75,1.5]",
                "metrics stratified by gold-known field count",
            ],
            "selection_use": "diagnostic only except the locked dominance gates",
        },
    }


def _uncertainty_contract() -> dict[str, Any]:
    return {
        "method": "paired product-group nonparametric percentile bootstrap",
        "resampling_unit": "entire k=8 product group",
        "completion_level_resampling": False,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_quantiles": [0.025, 0.975],
        "paired_deltas": (
            "resample one shared group-index vector and recompute candidate-minus-comparator"
        ),
        "primary_minimum_group_support": MIN_PRIMARY_GROUP_SUPPORT,
        "p_values": "not used for selection",
        "multiple_candidates": (
            "all three candidates and the selection hierarchy are predeclared; report every comparison"
        ),
    }


def _acceptance_gates(baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "G1_integrity_and_tests",
            "metric": "all lineage checks, deterministic rebuilds and CPU tests",
            "operator": "all_true",
            "threshold": True,
        },
        {
            "id": "G2_active_variation",
            "metric": "active zero-variance group share",
            "operator": "equals",
            "threshold": 0.0,
            "baseline": baseline["active_zero_variance_share"],
        },
        {
            "id": "G3_distinct_levels",
            "metric": "active groups with at least 3 distinct reward levels share",
            "operator": "greater_than_or_equal",
            "threshold": 0.50,
            "baseline": baseline["active_groups_with_at_least_three_levels_share"],
            "rationale": "move from a rare 5.3% exception to at least half of groups",
        },
        {
            "id": "G4_large_ties",
            "metric": "active groups whose largest reward tie contains at least 6 of 8 completions",
            "operator": "less_than_or_equal",
            "threshold": 0.50,
            "baseline": baseline["active_groups_with_large_tie_at_least_six_share"],
            "rationale": "reduce the original roughly two-thirds large-tie rate below one-half",
        },
        {
            "id": "G5_pairwise_resolution",
            "metric": "mean group pairwise discrimination candidate-minus-original",
            "operator": "point_at_least_and_ci_lower_above",
            "point_threshold": 0.10,
            "ci_lower_threshold": 0.0,
            "minimum_groups": MIN_PRIMARY_GROUP_SUPPORT,
        },
        {
            "id": "G6_known_utility_alignment",
            "metric": "mean group canonical-known-utility net alignment candidate-minus-original",
            "operator": "point_nonnegative_and_ci_lower_noninferior",
            "point_threshold": 0.0,
            "ci_lower_threshold": -0.02,
            "minimum_groups": MIN_PRIMARY_GROUP_SUPPORT,
        },
        {
            "id": "G7_harmful_coverage",
            "metric": "mean group harmful-coverage preference candidate-minus-original",
            "operator": "point_and_ci_upper_at_most",
            "point_threshold": 0.02,
            "ci_upper_threshold": 0.05,
            "minimum_groups": MIN_PRIMARY_GROUP_SUPPORT,
        },
        {
            "id": "G8_field_dominance",
            "metric": "largest field share of absolute semantic contribution",
            "operator": "less_than_or_equal",
            "threshold": 0.20,
            "rationale": "no field may contribute more than three times the uniform 1/15 share",
        },
        {
            "id": "G9_cb_class_dominance",
            "applies_to": ["CB"],
            "metric": "largest gold-class share of absolute known semantic contribution",
            "operator": "less_than_or_equal",
            "threshold": 0.15,
        },
        {
            "id": "G10_full_training_variation",
            "metric": "full authoritative-training zero-variance group share",
            "operator": "less_than_or_equal",
            "threshold": 0.40,
            "baseline": baseline["full_training_zero_variance_share"],
            "rationale": "require an 8.5-point practical reduction from the locked 48.5% baseline",
            "requires_separate_full_scope_raw_replay": True,
        },
    ]


def _selection_hierarchy() -> dict[str, Any]:
    return {
        "principle": "choose the simplest passing policy unless extra semantics earn their complexity",
        "steps": [
            {
                "order": 1,
                "action": "discard every candidate that fails any applicable universal gate",
            },
            {
                "order": 2,
                "comparison": "UA versus U",
                "choose_more_complex_if": {
                    "unknown_abstention_net_alignment_point_delta_minimum": 0.10,
                    "unknown_abstention_net_alignment_ci_lower_minimum": 0.0,
                    "known_utility_alignment_ci_lower_noninferiority": -0.02,
                    "pairwise_discrimination_ci_lower_noninferiority": -0.02,
                    "harmful_coverage_ci_upper_noninferiority": 0.02,
                },
                "otherwise": "prefer U",
            },
            {
                "order": 3,
                "comparison": "CB versus the surviving uniform candidate",
                "choose_more_complex_if": {
                    "class_balanced_known_utility_alignment_point_delta_minimum": 0.03,
                    "class_balanced_known_utility_alignment_ci_lower_minimum": 0.0,
                    "canonical_known_utility_alignment_ci_lower_noninferiority": -0.02,
                    "pairwise_discrimination_ci_lower_noninferiority": -0.02,
                    "harmful_coverage_ci_upper_noninferiority": 0.02,
                    "field_and_class_dominance_gates_pass": True,
                },
                "otherwise": "prefer the simpler surviving uniform candidate",
            },
        ],
        "unresolved_tie": "prefer U over UA over CB",
        "no_passing_candidate": "stop Phase D; do not dispatch GPU training",
        "winner_claim": "offline reward-design selection only, not evidence of policy improvement",
    }


def build_contract(
    *,
    repo_root: str | Path,
    document_path: str | Path = DEFAULT_DOCUMENT,
    payoff_contract_path: str | Path = DEFAULT_PAYOFF_CONTRACT,
    scale_contract_path: str | Path = DEFAULT_SCALE_CONTRACT,
    original_replay_path: str | Path = DEFAULT_ORIGINAL_REPLAY,
    candidate_manifest_path: str | Path = DEFAULT_CANDIDATE_MANIFEST,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    document = _resolve(repo_root, document_path)
    payoff = _resolve(repo_root, payoff_contract_path)
    scale = _resolve(repo_root, scale_contract_path)
    original_path = _resolve(repo_root, original_replay_path)
    candidate_manifest_path = _resolve(repo_root, candidate_manifest_path)

    if VERSION not in document.read_text(encoding="utf-8"):
        raise RuntimeError("comparison document does not declare the expected version")
    original = _load_json(original_path)
    candidate_manifest = _load_json(candidate_manifest_path)
    if original.get("version") != EXPECTED_ORIGINAL_VERSION:
        raise RuntimeError("unexpected original reward replay version")
    if candidate_manifest.get("version") != EXPECTED_CANDIDATE_REPLAY_VERSION:
        raise RuntimeError("unexpected candidate replay manifest version")
    if tuple(candidate_manifest["record_contract"]["candidate_order"]) != EXPECTED_CANDIDATES:
        raise RuntimeError("candidate order drifted")
    output = candidate_manifest["output"]
    if (
        output.get("jsonl_group_records") != EXPECTED_ACTIVE_GROUPS
        or output.get("completion_records") != EXPECTED_ACTIVE_COMPLETIONS
    ):
        raise RuntimeError("candidate replay manifest counts drifted")
    boundary = candidate_manifest["selection_boundary"]
    forbidden_true = (
        boundary.get("aggregate_candidate_comparison_calculated"),
        boundary.get("candidate_rankings_calculated"),
        boundary.get("acceptance_thresholds_applied"),
        boundary.get("winner_selected"),
    )
    if any(forbidden_true):
        raise RuntimeError("candidate outcomes were already aggregated or selected")

    baseline = _baseline(original)
    implementation = Path(__file__).resolve()
    return {
        "version": VERSION,
        "status": "locked_before_candidate_aggregation",
        "role": "predeclared D3 metrics, D4 gates and candidate selection hierarchy",
        "selection_boundary": {
            "allowed": "locked training-only original baseline and raw candidate replay manifest metadata",
            "prohibited": [
                "candidate replay record contents or aggregate candidate outcomes",
                "SFT validation", "legacy frozen 300", "probe 100",
                "future confirmation data",
            ],
            "candidate_replay_records_opened": False,
            "candidate_aggregate_metrics_calculated": False,
            "candidate_rankings_calculated": False,
            "acceptance_gates_applied": False,
            "winner_selected": False,
            "cuda_imports_performed": False,
        },
        "inputs": {
            "comparison_document": _file_metadata(document, repo_root),
            "ordinal_payoff_contract": _file_metadata(payoff, repo_root),
            "numeric_scale_contract": _file_metadata(scale, repo_root),
            "original_reward_replay": _file_metadata(original_path, repo_root),
            "candidate_replay_manifest": _file_metadata(candidate_manifest_path, repo_root),
            "candidate_replay_records_identity_from_manifest_only": {
                "path": output["path"],
                "bytes": output["bytes"],
                "sha256": output["sha256"],
                "opened_by_contract_builder": False,
            },
        },
        "implementation": _file_metadata(implementation, repo_root),
        "lineage": {
            "active_groups": EXPECTED_ACTIVE_GROUPS,
            "group_size": EXPECTED_GROUP_SIZE,
            "active_completions": EXPECTED_ACTIVE_COMPLETIONS,
            "ordered_sku_sha256": output["ordered_sku_sha256"],
            "ordered_rollout_key_sha256": output["ordered_rollout_key_sha256"],
            "same_order_as_original_active_scope": (
                output["ordered_sku_sha256"]
                == original["scopes"]["run2_active_pool"]["ordered_sku_sha256"]
                and output["ordered_rollout_key_sha256"]
                == original["scopes"]["run2_active_pool"]["ordered_rollout_key_sha256"]
            ),
        },
        "locked_original_baseline": baseline,
        "metric_contract": _metric_contract(),
        "uncertainty_contract": _uncertainty_contract(),
        "universal_acceptance_gates": _acceptance_gates(baseline),
        "selection_hierarchy": _selection_hierarchy(),
        "next_step": (
            "implement the D3 aggregator against these locked definitions, then open the raw "
            "candidate replay records once and publish all candidate results together"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--payoff-contract", default=DEFAULT_PAYOFF_CONTRACT)
    parser.add_argument("--scale-contract", default=DEFAULT_SCALE_CONTRACT)
    parser.add_argument("--original-replay", default=DEFAULT_ORIGINAL_REPLAY)
    parser.add_argument("--candidate-manifest", default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output = _resolve(repo_root, args.output)
    contract = build_contract(
        repo_root=repo_root,
        document_path=args.document,
        payoff_contract_path=args.payoff_contract,
        scale_contract_path=args.scale_contract,
        original_replay_path=args.original_replay,
        candidate_manifest_path=args.candidate_manifest,
    )
    write_exclusive_atomic_json(output, contract)
    print(json.dumps({"output": str(output), "status": contract["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
