#!/usr/bin/env python3
"""Production result contract for the Run 2 active-pool D3 analysis.

This module does not open replay evidence or calculate aggregate metrics. It
validates an already-built production analysis artifact and publishes it only
to the locked D3 path using exclusive atomic creation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from training.analyze_run2_candidates import (
    REWARD_LABELS,
    VERSION as ANALYSIS_CORE_VERSION,
)
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.run2_analysis_orchestrator import (
    PRODUCTION_ROLE,
    VERSION as ORCHESTRATOR_VERSION,
)
from training.run2_comparison_contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    EXPECTED_ACTIVE_COMPLETIONS,
    EXPECTED_ACTIVE_GROUPS,
    EXPECTED_CANDIDATES,
    VERSION as COMPARISON_CONTRACT_VERSION,
)
from training.run2_segment_summaries import (
    CB_CLASS_DOMINANCE_THRESHOLD_REFERENCE,
    FIELD_DOMINANCE_THRESHOLD_REFERENCE,
    VERSION as DIAGNOSTICS_VERSION,
)


VERSION = "grpo-run2-d3-production-result-contract-v1"
DEFAULT_OUTPUT = "runs/grpo-run2-d3-candidate-analysis.json"
EXPECTED_ACTIVE_MANIFEST = MappingProxyType({
    "path": "runs/grpo-run2-candidate-replay-manifest.json",
    "bytes": 6_435,
    "sha256": "e10c3c47bb54fe0ad4bd07e68966401be71771d2bf134aa2b29dbb9c1683163e",
})
EXPECTED_ACTIVE_RECORDS = MappingProxyType({
    "path": "runs/grpo-run2-candidate-replay-records.jsonl.gz",
    "bytes": 1_921_202,
    "sha256": "30e3ea8681ca80de5737cc5928b8a755e8b14cd36f6c0a67e00385a8408be38a",
})
EXPECTED_COMPARISON_CONTRACT = MappingProxyType({
    "path": "runs/grpo-run2-comparison-contract.json",
    "bytes": 12_079,
    "sha256": "8692291af2319c33a9a6548c1a6530f8c61da0c04e27e69eaa048584245e1142",
})
EXPECTED_CLASS_WEIGHTS = MappingProxyType({
    "path": "runs/grpo-run2-cb-class-weights.json",
    "bytes": 27_446,
    "sha256": "7b53323a7f1c170fa68c6b1a0d1356c67fd827f70f466ba2972b857418f4ab37",
})
EXPECTED_ORDERED_SKU_SHA256 = (
    "97a96e775f2644c35cd33a412113b9bb135fba5019c5803e58f05ebd954eb4c7"
)
EXPECTED_ORDERED_ROLLOUT_KEY_SHA256 = (
    "c1fe09c88fa4a09e2397d6a98b4cd400dfdb57a9ef71cf61190ae0b96aef5500"
)
EXPECTED_TARGETS = (
    "known_exact_rate",
    "known_coverage",
    "selective_correctness",
    "unknown_abstention_rate",
    "rule_quality",
    "canonical_known_utility",
    "class_balanced_known_utility",
)
EXPECTED_PAIRED_METRICS = (
    "pairwise_discrimination",
    "canonical_known_utility_net_alignment",
    "harmful_coverage_preference",
)
EXPECTED_SEGMENT_DIMENSIONS = (
    "product_category",
    "difficulty_band",
    "gold_known_field_count",
)

EXPECTED_D3_PREFLIGHT_BOUNDARY = MappingProxyType({
    "replay_gzip_decompressed": False,
    "replay_records_parsed": False,
    "candidate_aggregate_metrics_calculated": False,
    "acceptance_gates_applied": False,
    "candidate_rankings_calculated": False,
    "winner_selected": False,
    "artifact_published": False,
})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("D3 artifact must be finite canonical JSON") from exc


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} schema drifted")


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ordered_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_distribution(
    value: Any,
    *,
    expected_count: int,
    name: str,
) -> None:
    distribution = _mapping(value, name)
    _exact_keys(
        distribution,
        {
            "count",
            "minimum",
            "p05",
            "p25",
            "median",
            "mean",
            "p75",
            "p95",
            "maximum",
            "population_std",
            "histogram",
        },
        name,
    )
    if distribution.get("count") != expected_count:
        raise ValueError(f"{name} count drifted")
    for key in (
        "minimum",
        "p05",
        "p25",
        "median",
        "mean",
        "p75",
        "p95",
        "maximum",
        "population_std",
    ):
        _finite(distribution.get(key), f"{name} {key}")
    histogram = _mapping(distribution.get("histogram"), f"{name} histogram")
    if not histogram or any(
        not isinstance(count, int) or isinstance(count, bool) or count <= 0
        for count in histogram.values()
    ):
        raise ValueError(f"{name} histogram is invalid")
    if sum(histogram.values()) != expected_count:
        raise ValueError(f"{name} histogram denominator drifted")


def _validate_reward_summary(
    value: Any,
    *,
    group_ids: set[str],
    name: str,
) -> None:
    summary = _mapping(value, name)
    _exact_keys(
        summary,
        {
            "groups",
            "completions",
            "completion_reward_distribution",
            "group_mean_distribution",
            "within_group_variance_own_scale_distribution",
            "pairwise_discrimination_distribution",
            "unique_reward_levels_per_group_histogram",
            "largest_tie_size_per_group_histogram",
            "zero_variance_groups",
            "zero_variance_share",
            "groups_with_at_least_three_levels",
            "groups_with_at_least_three_levels_share",
            "groups_with_largest_tie_at_least_six",
            "groups_with_largest_tie_at_least_six_share",
            "group_values_for_paired_analysis",
        },
        name,
    )
    if summary.get("groups") != EXPECTED_ACTIVE_GROUPS:
        raise ValueError(f"{name} group denominator drifted")
    if summary.get("completions") != EXPECTED_ACTIVE_COMPLETIONS:
        raise ValueError(f"{name} completion denominator drifted")
    _validate_distribution(
        summary.get("completion_reward_distribution"),
        expected_count=EXPECTED_ACTIVE_COMPLETIONS,
        name=f"{name} completion distribution",
    )
    for key in (
        "group_mean_distribution",
        "within_group_variance_own_scale_distribution",
        "pairwise_discrimination_distribution",
    ):
        _validate_distribution(
            summary.get(key),
            expected_count=EXPECTED_ACTIVE_GROUPS,
            name=f"{name} {key}",
        )

    unique_histogram = _mapping(
        summary.get("unique_reward_levels_per_group_histogram"),
        f"{name} unique-level histogram",
    )
    largest_tie_histogram = _mapping(
        summary.get("largest_tie_size_per_group_histogram"),
        f"{name} largest-tie histogram",
    )
    for histogram, histogram_name in (
        (unique_histogram, "unique-level"),
        (largest_tie_histogram, "largest-tie"),
    ):
        counts = {}
        for raw_key, count in histogram.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} {histogram_name} key is invalid") from exc
            if str(key) != raw_key or not 1 <= key <= 8:
                raise ValueError(f"{name} {histogram_name} key is invalid")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(f"{name} {histogram_name} count is invalid")
            counts[key] = count
        if sum(counts.values()) != EXPECTED_ACTIVE_GROUPS:
            raise ValueError(f"{name} {histogram_name} denominator drifted")
        if histogram_name == "unique-level":
            if summary.get("zero_variance_groups") != counts.get(1, 0):
                raise ValueError(f"{name} zero-variance count drifted")
            if summary.get("groups_with_at_least_three_levels") != sum(
                count for level, count in counts.items() if level >= 3
            ):
                raise ValueError(f"{name} three-level count drifted")
        elif summary.get("groups_with_largest_tie_at_least_six") != sum(
            count for level, count in counts.items() if level >= 6
        ):
            raise ValueError(f"{name} largest-tie count drifted")

    for count_key, share_key in (
        ("zero_variance_groups", "zero_variance_share"),
        ("groups_with_at_least_three_levels", "groups_with_at_least_three_levels_share"),
        ("groups_with_largest_tie_at_least_six", "groups_with_largest_tie_at_least_six_share"),
    ):
        count = summary.get(count_key)
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(f"{name} {count_key} is invalid")
        expected_share = count / EXPECTED_ACTIVE_GROUPS
        if _finite(summary.get(share_key), f"{name} {share_key}") != expected_share:
            raise ValueError(f"{name} {share_key} drifted")

    group_values = _mapping(
        summary.get("group_values_for_paired_analysis"),
        f"{name} paired group values",
    )
    if set(group_values) != group_ids:
        raise ValueError(f"{name} paired group denominator drifted")
    for group_id, group_value in group_values.items():
        group_value = _mapping(group_value, f"{name} {group_id}")
        if set(group_value) != {"pairwise_discrimination_rate"}:
            raise ValueError(f"{name} paired group value schema drifted")
        rate = _finite(
            group_value.get("pairwise_discrimination_rate"),
            f"{name} pairwise discrimination",
        )
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} pairwise discrimination is out of range")


def _validate_directional_summary(
    value: Any,
    *,
    group_ids: set[str],
    name: str,
) -> None:
    summary = _mapping(value, name)
    _exact_keys(
        summary,
        {
            "groups_total",
            "groups_contributing",
            "comparable_pairs",
            "concordant_pairs",
            "discordant_pairs",
            "reward_ties",
            "group_net_alignment_distribution",
            "group_values_for_paired_analysis",
        },
        name,
    )
    if summary.get("groups_total") != EXPECTED_ACTIVE_GROUPS:
        raise ValueError(f"{name} total-group denominator drifted")
    contributing = summary.get("groups_contributing")
    if (
        not isinstance(contributing, int)
        or isinstance(contributing, bool)
        or not 0 <= contributing <= EXPECTED_ACTIVE_GROUPS
    ):
        raise ValueError(f"{name} contributing-group count is invalid")
    counts = [summary.get(key) for key in (
        "comparable_pairs",
        "concordant_pairs",
        "discordant_pairs",
        "reward_ties",
    )]
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
        raise ValueError(f"{name} pair counts are invalid")
    if counts[0] != counts[1] + counts[2] + counts[3]:
        raise ValueError(f"{name} pair counts do not reconcile")
    group_values = _mapping(
        summary.get("group_values_for_paired_analysis"),
        f"{name} paired values",
    )
    if len(group_values) != contributing or not set(group_values) <= group_ids:
        raise ValueError(f"{name} contributing-group denominator drifted")
    if contributing:
        _validate_distribution(
            summary.get("group_net_alignment_distribution"),
            expected_count=contributing,
            name=f"{name} alignment distribution",
        )
    elif summary.get("group_net_alignment_distribution") is not None:
        raise ValueError(f"{name} empty alignment distribution must be null")


def _validate_harmful_summary(
    value: Any,
    *,
    group_ids: set[str],
    name: str,
) -> None:
    summary = _mapping(value, name)
    _exact_keys(
        summary,
        {
            "groups_total",
            "groups_contributing",
            "comparable_pairs",
            "harmful_preferences",
            "safe_preferences",
            "reward_ties",
            "group_score_distribution",
            "group_values_for_paired_analysis",
        },
        name,
    )
    if summary.get("groups_total") != EXPECTED_ACTIVE_GROUPS:
        raise ValueError(f"{name} total-group denominator drifted")
    contributing = summary.get("groups_contributing")
    if (
        not isinstance(contributing, int)
        or isinstance(contributing, bool)
        or not 0 <= contributing <= EXPECTED_ACTIVE_GROUPS
    ):
        raise ValueError(f"{name} contributing-group count is invalid")
    comparable = summary.get("comparable_pairs")
    children = [summary.get(key) for key in (
        "harmful_preferences",
        "safe_preferences",
        "reward_ties",
    )]
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in [comparable, *children]):
        raise ValueError(f"{name} pair counts are invalid")
    if comparable != sum(children):
        raise ValueError(f"{name} pair counts do not reconcile")
    group_values = _mapping(
        summary.get("group_values_for_paired_analysis"),
        f"{name} paired values",
    )
    if len(group_values) != contributing or not set(group_values) <= group_ids:
        raise ValueError(f"{name} contributing-group denominator drifted")
    if contributing:
        _validate_distribution(
            summary.get("group_score_distribution"),
            expected_count=contributing,
            name=f"{name} score distribution",
        )
    elif summary.get("group_score_distribution") is not None:
        raise ValueError(f"{name} empty score distribution must be null")


def _validate_bootstrap(value: Any, *, name: str, require_all_groups: bool) -> None:
    bootstrap = _mapping(value, name)
    _exact_keys(
        bootstrap,
        {
            "method",
            "pairing_unit",
            "completion_level_resampling",
            "groups_per_replicate",
            "seed",
            "replicates",
            "confidence",
            "percentile_method",
            "replicate_stream_sha256",
            "baseline",
            "candidate",
            "delta_candidate_minus_baseline",
        },
        name,
    )
    if bootstrap.get("method") != (
        "paired nonparametric product-group bootstrap, percentile interval"
    ):
        raise ValueError(f"{name} method drifted")
    if bootstrap.get("pairing_unit") != "entire k=8 product group":
        raise ValueError(f"{name} pairing unit drifted")
    if bootstrap.get("completion_level_resampling") is not False:
        raise ValueError(f"{name} resampled completions")
    groups = bootstrap.get("groups_per_replicate")
    if not isinstance(groups, int) or isinstance(groups, bool) or not 1 <= groups <= EXPECTED_ACTIVE_GROUPS:
        raise ValueError(f"{name} bootstrap group denominator is invalid")
    if require_all_groups and groups != EXPECTED_ACTIVE_GROUPS:
        raise ValueError(f"{name} bootstrap omitted active groups")
    if bootstrap.get("seed") != BOOTSTRAP_SEED:
        raise ValueError(f"{name} bootstrap seed drifted")
    if bootstrap.get("replicates") != BOOTSTRAP_REPLICATES:
        raise ValueError(f"{name} bootstrap replicate count drifted")
    if bootstrap.get("confidence") != CONFIDENCE_LEVEL:
        raise ValueError(f"{name} bootstrap confidence drifted")
    if bootstrap.get("percentile_method") != (
        "linear interpolation at sorted index (n - 1) * p"
    ):
        raise ValueError(f"{name} percentile method drifted")
    stream_hash = bootstrap.get("replicate_stream_sha256")
    if not _is_sha256(stream_hash):
        raise ValueError(f"{name} replicate stream hash is invalid")
    for summary_name in ("baseline", "candidate"):
        summary = _mapping(bootstrap.get(summary_name), f"{name} {summary_name}")
        _exact_keys(summary, {"point", "mean", "median", "ci"}, f"{name} {summary_name}")
        for key in ("point", "mean", "median"):
            _finite(summary.get(key), f"{name} {summary_name} {key}")
        ci = summary.get("ci")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError(f"{name} {summary_name} CI is invalid")
        _finite(ci[0], f"{name} {summary_name} CI lower")
        _finite(ci[1], f"{name} {summary_name} CI upper")
    delta = _mapping(
        bootstrap.get("delta_candidate_minus_baseline"),
        f"{name} delta",
    )
    _exact_keys(
        delta,
        {
            "point",
            "mean",
            "median",
            "ci",
            "fraction_below_zero",
            "fraction_equal_zero",
            "fraction_above_zero",
        },
        f"{name} delta",
    )
    for key in ("point", "mean", "median"):
        _finite(delta.get(key), f"{name} delta {key}")
    ci = delta.get("ci")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{name} delta CI is invalid")
    _finite(ci[0], f"{name} delta CI lower")
    _finite(ci[1], f"{name} delta CI upper")
    fractions = [
        _finite(delta.get(key), f"{name} delta {key}")
        for key in (
            "fraction_below_zero",
            "fraction_equal_zero",
            "fraction_above_zero",
        )
    ]
    if any(not 0.0 <= fraction <= 1.0 for fraction in fractions) or not math.isclose(
        sum(fractions),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{name} delta fractions drifted")


def _validate_analysis_core(
    value: Any,
    *,
    lineage: Mapping[str, Any],
) -> set[str]:
    core = _mapping(value, "D3 analysis core")
    _exact_keys(
        core,
        {
            "version",
            "status",
            "boundary",
            "groups",
            "completions",
            "group_order",
            "reward_summaries",
            "directional_alignments",
            "harmful_coverage",
            "paired_candidate_minus_original",
        },
        "D3 analysis core",
    )
    if core.get("version") != ANALYSIS_CORE_VERSION or core.get("status") != "analysis_core_completed":
        raise ValueError("D3 analysis core identity drifted")
    boundary = _mapping(core.get("boundary"), "D3 analysis core boundary")
    expected_boundary = {
        "input_kind": "already-materialized in-memory observations",
        "file_io_performed": False,
        "real_candidate_replay_opened": False,
        "acceptance_gates_applied": False,
        "winner_selected": False,
    }
    if dict(boundary) != expected_boundary:
        raise ValueError("D3 analysis core boundary drifted")
    if core.get("groups") != EXPECTED_ACTIVE_GROUPS or core.get("completions") != EXPECTED_ACTIVE_COMPLETIONS:
        raise ValueError("D3 analysis core denominator drifted")
    group_order = core.get("group_order")
    if not isinstance(group_order, list) or len(group_order) != EXPECTED_ACTIVE_GROUPS:
        raise ValueError("D3 analysis core group order denominator drifted")
    if any(not isinstance(group_id, str) or not group_id for group_id in group_order):
        raise ValueError("D3 analysis core group ID is invalid")
    if len(set(group_order)) != EXPECTED_ACTIVE_GROUPS:
        raise ValueError("D3 analysis core group IDs are not unique")
    if _ordered_hash(group_order) != lineage["ordered_sku_sha256"]:
        raise ValueError("D3 analysis core ordered SKU hash drifted")
    group_ids = set(group_order)

    reward_summaries = _mapping(core.get("reward_summaries"), "D3 reward summaries")
    if set(reward_summaries) != set(REWARD_LABELS):
        raise ValueError("D3 reward summary set drifted")
    for reward in REWARD_LABELS:
        _validate_reward_summary(
            reward_summaries[reward],
            group_ids=group_ids,
            name=f"D3 reward summary {reward}",
        )

    alignments = _mapping(core.get("directional_alignments"), "D3 directional alignments")
    if set(alignments) != set(REWARD_LABELS):
        raise ValueError("D3 directional reward set drifted")
    for reward in REWARD_LABELS:
        targets = _mapping(alignments[reward], f"D3 {reward} alignments")
        if set(targets) != set(EXPECTED_TARGETS):
            raise ValueError(f"D3 {reward} target set drifted")
        for target in EXPECTED_TARGETS:
            _validate_directional_summary(
                targets[target],
                group_ids=group_ids,
                name=f"D3 {reward}/{target} alignment",
            )

    harmful = _mapping(core.get("harmful_coverage"), "D3 harmful coverage")
    if set(harmful) != set(REWARD_LABELS):
        raise ValueError("D3 harmful-coverage reward set drifted")
    for reward in REWARD_LABELS:
        _validate_harmful_summary(
            harmful[reward],
            group_ids=group_ids,
            name=f"D3 {reward} harmful coverage",
        )

    paired = _mapping(
        core.get("paired_candidate_minus_original"),
        "D3 paired candidate comparisons",
    )
    if set(paired) != set(EXPECTED_CANDIDATES):
        raise ValueError("D3 paired candidate set drifted")
    for candidate in EXPECTED_CANDIDATES:
        metrics = _mapping(paired[candidate], f"D3 paired {candidate}")
        if set(metrics) != set(EXPECTED_PAIRED_METRICS):
            raise ValueError(f"D3 paired {candidate} metric set drifted")
        for metric in EXPECTED_PAIRED_METRICS:
            result = metrics[metric]
            if result is None:
                if metric == "pairwise_discrimination":
                    raise ValueError("D3 pairwise discrimination bootstrap is missing")
                continue
            _validate_bootstrap(
                result,
                name=f"D3 {candidate}/{metric} bootstrap",
                require_all_groups=metric == "pairwise_discrimination",
            )
    return group_ids


def _validate_diagnostics(value: Any, *, group_ids: set[str]) -> None:
    diagnostics = _mapping(value, "D3 diagnostics")
    _exact_keys(
        diagnostics,
        {"version", "status", "boundary", "dominance", "product_segments"},
        "D3 diagnostics",
    )
    if diagnostics.get("version") != DIAGNOSTICS_VERSION or diagnostics.get("status") != "in_memory_diagnostics_completed":
        raise ValueError("D3 diagnostics identity drifted")
    expected_boundary = {
        "file_io_performed": False,
        "real_candidate_replay_opened_by_this_module": False,
        "acceptance_gates_applied": False,
        "winner_selected": False,
    }
    if dict(_mapping(diagnostics.get("boundary"), "D3 diagnostics boundary")) != expected_boundary:
        raise ValueError("D3 diagnostics boundary drifted")

    dominance = _mapping(diagnostics.get("dominance"), "D3 dominance")
    if dominance.get("version") != DIAGNOSTICS_VERSION or dominance.get("groups") != EXPECTED_ACTIVE_GROUPS:
        raise ValueError("D3 dominance denominator drifted")
    fields = _mapping(dominance.get("field_contributions"), "D3 field contributions")
    if set(fields) != set(EXPECTED_CANDIDATES):
        raise ValueError("D3 field-contribution candidate set drifted")
    for candidate in EXPECTED_CANDIDATES:
        summary = _mapping(fields[candidate], f"D3 {candidate} field contribution")
        if summary.get("dominance_threshold_reference") != FIELD_DOMINANCE_THRESHOLD_REFERENCE:
            raise ValueError(f"D3 {candidate} field threshold drifted")
        if summary.get("dominance_gate_applied") is not False:
            raise ValueError(f"D3 {candidate} field gate was applied")
    classes = _mapping(dominance.get("cb_class_contributions"), "D3 CB classes")
    if classes.get("dominance_threshold_reference") != CB_CLASS_DOMINANCE_THRESHOLD_REFERENCE:
        raise ValueError("D3 CB class threshold drifted")
    if classes.get("dominance_gate_applied") is not False:
        raise ValueError("D3 CB class gate was applied")
    guardrails = _mapping(dominance.get("grain_guardrails"), "D3 grain guardrails")
    expected_guardrails = {
        "product_counts_derived_from_group_observations_only": True,
        "field_child_rows_used_as_product_denominator": False,
        "class_child_rows_used_as_product_denominator": False,
        "class_allocations_reconstruct_cb_known_fields": True,
    }
    if dict(guardrails) != expected_guardrails:
        raise ValueError("D3 grain guardrails drifted")

    segments = _mapping(diagnostics.get("product_segments"), "D3 product segments")
    if segments.get("version") != DIAGNOSTICS_VERSION or segments.get("groups") != EXPECTED_ACTIVE_GROUPS:
        raise ValueError("D3 product-segment denominator drifted")
    for key in (
        "segment_keys_are_pre_outcome",
    ):
        if segments.get(key) is not True:
            raise ValueError(f"D3 product segments {key} drifted")
    for key in ("bootstrap_performed", "acceptance_gates_applied", "winner_selected"):
        if segments.get(key) is not False:
            raise ValueError(f"D3 product segments {key} must be false")
    dimensions = _mapping(segments.get("dimensions"), "D3 segment dimensions")
    if set(dimensions) != set(EXPECTED_SEGMENT_DIMENSIONS):
        raise ValueError("D3 segment dimension set drifted")
    for dimension in EXPECTED_SEGMENT_DIMENSIONS:
        summary = _mapping(dimensions[dimension], f"D3 segment {dimension}")
        members = _mapping(summary.get("segments"), f"D3 {dimension} members")
        if summary.get("segment_count") != len(members):
            raise ValueError(f"D3 {dimension} segment count drifted")
        if summary.get("group_memberships") != EXPECTED_ACTIVE_GROUPS:
            raise ValueError(f"D3 {dimension} membership denominator drifted")
        if summary.get("every_group_appears_exactly_once") is not True:
            raise ValueError(f"D3 {dimension} membership is not one-to-one")
        if sum(
            _mapping(segment, f"D3 {dimension} member").get("groups", -1)
            for segment in members.values()
        ) != EXPECTED_ACTIVE_GROUPS:
            raise ValueError(f"D3 {dimension} segment subtotals drifted")
        observed_group_ids: list[str] = []
        for segment_name, raw_segment in members.items():
            segment = _mapping(raw_segment, f"D3 {dimension}/{segment_name}")
            _exact_keys(
                segment,
                {
                    "groups",
                    "completions",
                    "ordered_group_ids",
                    "ordered_group_id_sha256",
                    "interpretation_allowed",
                    "minimum_groups_for_interpretation",
                    "reward_summaries",
                    "directional_alignments",
                    "harmful_coverage",
                },
                f"D3 {dimension}/{segment_name}",
            )
            segment_groups = segment.get("groups")
            ordered_ids = segment.get("ordered_group_ids")
            if (
                not isinstance(segment_groups, int)
                or isinstance(segment_groups, bool)
                or segment_groups <= 0
                or not isinstance(ordered_ids, list)
                or len(ordered_ids) != segment_groups
            ):
                raise ValueError(f"D3 {dimension}/{segment_name} denominator drifted")
            if segment.get("completions") != segment_groups * 8:
                raise ValueError(f"D3 {dimension}/{segment_name} completions drifted")
            if len(set(ordered_ids)) != segment_groups or not set(ordered_ids) <= group_ids:
                raise ValueError(f"D3 {dimension}/{segment_name} group IDs drifted")
            if segment.get("ordered_group_id_sha256") != _ordered_hash(ordered_ids):
                raise ValueError(f"D3 {dimension}/{segment_name} group hash drifted")
            if segment.get("minimum_groups_for_interpretation") != 30:
                raise ValueError(f"D3 {dimension}/{segment_name} support rule drifted")
            if segment.get("interpretation_allowed") is not (segment_groups >= 30):
                raise ValueError(f"D3 {dimension}/{segment_name} interpretation drifted")
            if set(_mapping(segment.get("reward_summaries"), "segment rewards")) != set(REWARD_LABELS):
                raise ValueError(f"D3 {dimension}/{segment_name} reward set drifted")
            if set(_mapping(segment.get("directional_alignments"), "segment alignments")) != set(REWARD_LABELS):
                raise ValueError(f"D3 {dimension}/{segment_name} alignment set drifted")
            if set(_mapping(segment.get("harmful_coverage"), "segment harmful coverage")) != set(REWARD_LABELS):
                raise ValueError(f"D3 {dimension}/{segment_name} harmful set drifted")
            observed_group_ids.extend(ordered_ids)
        if len(observed_group_ids) != EXPECTED_ACTIVE_GROUPS or set(observed_group_ids) != group_ids:
            raise ValueError(f"D3 {dimension} segments do not partition active groups")


def validate_production_d3_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the production active-scope preflight without replay I/O."""
    report = _canonical_copy(_mapping(value, "D3 preflight"))
    _exact_keys(
        report,
        {
            "version",
            "status",
            "mode",
            "role",
            "inputs",
            "lineage",
            "settings",
            "comparison_contract_version",
            "selection_boundary",
        },
        "D3 preflight",
    )
    expected_identity = {
        "version": ORCHESTRATOR_VERSION,
        "status": "production_active_preflight_passed",
        "mode": "locked_active_replay",
        "role": PRODUCTION_ROLE,
        "comparison_contract_version": COMPARISON_CONTRACT_VERSION,
    }
    for key, expected in expected_identity.items():
        if report.get(key) != expected:
            raise ValueError(f"D3 preflight {key} drifted")
    expected_inputs = {
        "manifest": EXPECTED_ACTIVE_MANIFEST,
        "records": EXPECTED_ACTIVE_RECORDS,
        "comparison_contract": EXPECTED_COMPARISON_CONTRACT,
        "class_weights": EXPECTED_CLASS_WEIGHTS,
        "all_identities_verified_before_gzip_open": True,
    }
    if dict(_mapping(report.get("inputs"), "D3 preflight inputs")) != expected_inputs:
        raise ValueError("D3 preflight input identities drifted")
    expected_lineage = {
        "groups": EXPECTED_ACTIVE_GROUPS,
        "completions": EXPECTED_ACTIVE_COMPLETIONS,
        "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
        "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
        "contract_lineage_matches_manifest": True,
    }
    if dict(_mapping(report.get("lineage"), "D3 preflight lineage")) != expected_lineage:
        raise ValueError("D3 preflight lineage drifted")
    expected_settings = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence": CONFIDENCE_LEVEL,
        "settings_locked_to_production_contract": True,
    }
    if dict(_mapping(report.get("settings"), "D3 preflight settings")) != expected_settings:
        raise ValueError("D3 preflight settings drifted")
    if dict(
        _mapping(report.get("selection_boundary"), "D3 preflight boundary")
    ) != EXPECTED_D3_PREFLIGHT_BOUNDARY:
        raise ValueError("D3 preflight selection boundary drifted")
    return report


def validate_production_d3_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete production D3 artifact before publication."""
    artifact = _canonical_copy(_mapping(value, "D3 artifact"))
    _exact_keys(
        artifact,
        {
            "version",
            "status",
            "role",
            "mode",
            "selection_boundary",
            "inputs",
            "lineage",
            "settings",
            "implementation",
            "comparison_contract_version",
            "analysis_core",
            "contribution_and_segment_diagnostics",
        },
        "D3 artifact",
    )
    expected_identity = {
        "version": ORCHESTRATOR_VERSION,
        "status": "aggregate_candidate_analysis_completed_pending_gates",
        "role": PRODUCTION_ROLE,
        "mode": "locked_active_replay",
        "comparison_contract_version": COMPARISON_CONTRACT_VERSION,
    }
    for key, expected in expected_identity.items():
        if artifact.get(key) != expected:
            raise ValueError(f"D3 artifact {key} drifted")
    expected_boundary = {
        "candidate_aggregate_metrics_calculated": True,
        "real_candidate_replay_used": True,
        "acceptance_gates_applied": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
    }
    if dict(_mapping(artifact.get("selection_boundary"), "D3 selection boundary")) != expected_boundary:
        raise ValueError("D3 selection boundary drifted")

    inputs = _mapping(artifact.get("inputs"), "D3 inputs")
    expected_inputs = {
        "manifest": EXPECTED_ACTIVE_MANIFEST,
        "records": EXPECTED_ACTIVE_RECORDS,
        "comparison_contract": EXPECTED_COMPARISON_CONTRACT,
        "class_weights": EXPECTED_CLASS_WEIGHTS,
        "all_identities_verified_before_gzip_open": True,
    }
    if dict(inputs) != expected_inputs:
        raise ValueError("D3 input identities drifted")
    lineage = _mapping(artifact.get("lineage"), "D3 lineage")
    expected_lineage = {
        "groups": EXPECTED_ACTIVE_GROUPS,
        "completions": EXPECTED_ACTIVE_COMPLETIONS,
        "unique_skus": EXPECTED_ACTIVE_GROUPS,
        "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
        "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
        "groups_streamed_once": True,
        "groups_adapted_once": True,
    }
    if dict(lineage) != expected_lineage:
        raise ValueError("D3 lineage drifted")
    expected_settings = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence": CONFIDENCE_LEVEL,
        "settings_locked_to_production_contract": True,
    }
    if dict(_mapping(artifact.get("settings"), "D3 settings")) != expected_settings:
        raise ValueError("D3 settings drifted")
    implementation = _mapping(artifact.get("implementation"), "D3 implementation")
    if set(implementation) != {"path", "bytes", "sha256"}:
        raise ValueError("D3 implementation schema drifted")
    if implementation.get("path") != "training/run2_analysis_orchestrator.py":
        raise ValueError("D3 implementation path drifted")
    if not isinstance(implementation.get("bytes"), int) or implementation["bytes"] <= 0:
        raise ValueError("D3 implementation byte count is invalid")
    implementation_hash = implementation.get("sha256")
    if not _is_sha256(implementation_hash):
        raise ValueError("D3 implementation SHA-256 is invalid")

    group_ids = _validate_analysis_core(
        artifact.get("analysis_core"),
        lineage=lineage,
    )
    _validate_diagnostics(
        artifact.get("contribution_and_segment_diagnostics"),
        group_ids=group_ids,
    )
    return artifact


def publish_production_d3_result(
    *,
    repo_root: str | Path,
    artifact: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate and publish only to the locked D3 result path."""
    root = Path(repo_root).resolve()
    locked = (root / DEFAULT_OUTPUT).resolve()
    requested = Path(output_path)
    requested = (
        requested.resolve()
        if requested.is_absolute()
        else (root / requested).resolve()
    )
    if requested != locked:
        raise ValueError(f"D3 output path must be exactly {DEFAULT_OUTPUT}")
    checked = validate_production_d3_result(artifact)
    write_exclusive_atomic_json(locked, checked)
    return {
        "path": DEFAULT_OUTPUT,
        "bytes": locked.stat().st_size,
        "sha256": sha256_file(locked),
        "published_exclusively": True,
        "artifact": checked,
    }
