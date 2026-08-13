#!/usr/bin/env python3
"""Production result contract and atomic publisher for GRPO Run 2 Gate G10.

This module never opens replay evidence or calculates rewards. It validates a
production preflight plus an already-calculated in-memory collection, builds the
single-purpose Gate G10 result schema, and publishes only to one locked path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.replay_run2_full_training_candidates import (
    EXPECTED_COMPLETIONS,
    EXPECTED_TRAINING_GROUPS,
    FULL_TRAINING_ROLE,
)
from training.run2_analysis_orchestrator import (
    EXPECTED_FULL_MANIFEST_BYTES,
    EXPECTED_FULL_MANIFEST_SHA256,
    EXPECTED_FULL_RECORDS_BYTES,
    EXPECTED_FULL_RECORDS_SHA256,
    PRODUCTION_ROLE,
    VERSION as PREFLIGHT_VERSION,
)
from training.run2_comparison_contract import (
    COMPARISON_DECIMALS,
    EXPECTED_CANDIDATES,
    VERSION as COMPARISON_CONTRACT_VERSION,
)
from training.run2_gate_g10 import (
    GATE_ID,
    ORIGINAL_ZERO_VARIANCE_GROUPS,
    ORIGINAL_ZERO_VARIANCE_SHARE,
    THRESHOLD,
    THRESHOLD_DENOMINATOR,
    THRESHOLD_NUMERATOR,
    VERSION as GATE_CORE_VERSION,
)
from training.run2_gate_g10_collector import VERSION as COLLECTOR_VERSION


VERSION = "grpo-run2-gate-g10-production-result-v1"
ROLE = "full_authoritative_training_candidate_zero_variance_gate_result"
DEFAULT_OUTPUT = "runs/grpo-run2-gate-g10-result.json"
EXPECTED_ORDERED_SKU_SHA256 = (
    "05e22c09120a63f9936473fd1adf8bf7639545cbe2a22bdbb28b8ab2d74906ee"
)
EXPECTED_ORDERED_ROLLOUT_KEY_SHA256 = (
    "a6acc3446db2102b95fdec7fc798f731969a473b053bc23fe0e5a7a1d9851d59"
)
EXPECTED_ACTIVE_MANIFEST_SHA256 = (
    "e10c3c47bb54fe0ad4bd07e68966401be71771d2bf134aa2b29dbb9c1683163e"
)
EXPECTED_ACTIVE_RECORDS_SHA256 = (
    "30e3ea8681ca80de5737cc5928b8a755e8b14cd36f6c0a67e00385a8408be38a"
)
EXPECTED_COMPARISON_CONTRACT_SHA256 = (
    "8692291af2319c33a9a6548c1a6530f8c61da0c04e27e69eaa048584245e1142"
)
EXPECTED_CLASS_WEIGHTS_SHA256 = (
    "7b53323a7f1c170fa68c6b1a0d1356c67fd827f70f466ba2972b857418f4ab37"
)
EXPECTED_CB_EXTENSION_LEDGER_SHA256 = (
    "aeb089a1081d7efd1a99ccb2124e7b7412ec71f2362509f3df20dc2aa5837416"
)
MAXIMUM_ALLOWED_ZERO_VARIANCE_GROUPS = (
    EXPECTED_TRAINING_GROUPS
    * THRESHOLD_NUMERATOR
    // THRESHOLD_DENOMINATOR
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _canonical_copy(value: Mapping[str, Any], name: str) -> dict[str, Any]:
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
        raise ValueError(f"{name} is not finite canonical JSON data") from exc


def _require_false(mapping: Mapping[str, Any], keys: tuple[str, ...], name: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise ValueError(f"{name} {key} must be explicitly false")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_file_identity(
    metadata: Mapping[str, Any],
    *,
    name: str,
    expected_bytes: int | None,
    expected_sha256: str,
) -> dict[str, Any]:
    metadata = _mapping(metadata, name)
    observed_bytes = metadata.get("bytes")
    observed_sha256 = metadata.get("sha256")
    if expected_bytes is not None and observed_bytes != expected_bytes:
        raise ValueError(f"{name} byte count drifted")
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{name} SHA-256 drifted")
    path = metadata.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} path is invalid")
    return {
        "path": path,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def _validate_production_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    preflight = _mapping(preflight, "production preflight")
    if preflight.get("version") != PREFLIGHT_VERSION:
        raise ValueError("production preflight version drifted")
    if preflight.get("status") != "production_preflight_passed":
        raise ValueError("production preflight did not pass")
    if preflight.get("mode") != "locked_dual_replay":
        raise ValueError("production preflight mode drifted")
    if preflight.get("role") != PRODUCTION_ROLE:
        raise ValueError("production active replay role drifted")
    if preflight.get("comparison_contract_version") != COMPARISON_CONTRACT_VERSION:
        raise ValueError("production comparison contract version drifted")

    boundary = _mapping(
        preflight.get("selection_boundary"),
        "production preflight selection boundary",
    )
    _require_false(
        boundary,
        (
            "replay_gzip_decompressed",
            "replay_records_parsed",
            "full_replay_gzip_decompressed",
            "full_replay_records_parsed",
            "candidate_aggregate_metrics_calculated",
            "gate_g10_calculated",
            "acceptance_gates_applied",
            "candidate_rankings_calculated",
            "winner_selected",
            "artifact_published",
        ),
        "production preflight boundary",
    )
    inputs = _mapping(preflight.get("inputs"), "production preflight inputs")
    if inputs.get("all_identities_verified_before_gzip_open") is not True:
        raise ValueError("production preflight did not verify identities first")
    active_manifest = _verify_file_identity(
        inputs.get("manifest"),
        name="active replay manifest",
        expected_bytes=6_435,
        expected_sha256=EXPECTED_ACTIVE_MANIFEST_SHA256,
    )
    active_records = _verify_file_identity(
        inputs.get("records"),
        name="active replay records",
        expected_bytes=1_921_202,
        expected_sha256=EXPECTED_ACTIVE_RECORDS_SHA256,
    )
    comparison_contract = _verify_file_identity(
        inputs.get("comparison_contract"),
        name="comparison contract",
        expected_bytes=12_079,
        expected_sha256=EXPECTED_COMPARISON_CONTRACT_SHA256,
    )
    class_weights = _verify_file_identity(
        inputs.get("class_weights"),
        name="class weights",
        expected_bytes=27_446,
        expected_sha256=EXPECTED_CLASS_WEIGHTS_SHA256,
    )

    full = _mapping(
        preflight.get("full_training_replay"),
        "production full-training replay",
    )
    if full.get("role") != FULL_TRAINING_ROLE:
        raise ValueError("production full-training role drifted")
    expected_full_lineage = {
        "groups": EXPECTED_TRAINING_GROUPS,
        "completions": EXPECTED_COMPLETIONS,
        "active_groups_included": 1_438,
        "additional_training_groups": 1_802,
        "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
        "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
        "scope_is_distinct_from_active_replay": True,
        "all_identities_verified_before_gzip_open": True,
    }
    for key, expected in expected_full_lineage.items():
        if full.get(key) != expected:
            raise ValueError(f"production full-training {key} drifted")
    full_manifest = _verify_file_identity(
        full.get("manifest"),
        name="full-training replay manifest",
        expected_bytes=EXPECTED_FULL_MANIFEST_BYTES,
        expected_sha256=EXPECTED_FULL_MANIFEST_SHA256,
    )
    full_records = _verify_file_identity(
        full.get("records"),
        name="full-training replay records",
        expected_bytes=EXPECTED_FULL_RECORDS_BYTES,
        expected_sha256=EXPECTED_FULL_RECORDS_SHA256,
    )
    cb_extension = _mapping(
        full.get("cb_extension"),
        "production full-training CB extension",
    )
    if cb_extension.get("ordered_entry_ledger_sha256") != (
        EXPECTED_CB_EXTENSION_LEDGER_SHA256
    ):
        raise ValueError("production CB extension ledger SHA-256 drifted")
    if cb_extension.get("ledger_hash_recomputed") is not True:
        raise ValueError("production CB extension ledger was not recomputed")
    if cb_extension.get("active_weights_changed") is not False:
        raise ValueError("production CB extension changed active weights")
    if cb_extension.get("candidate_aggregates_calculated") is not False:
        raise ValueError("production CB extension used candidate aggregates")

    settings = _mapping(preflight.get("settings"), "production preflight settings")
    if settings.get("settings_locked_to_production_contract") is not True:
        raise ValueError("production settings are not locked")
    return {
        "preflight_version": preflight["version"],
        "active_manifest": active_manifest,
        "active_records": active_records,
        "comparison_contract": comparison_contract,
        "class_weights": class_weights,
        "full_manifest": full_manifest,
        "full_records": full_records,
        "cb_extension_ledger_sha256": EXPECTED_CB_EXTENSION_LEDGER_SHA256,
        "all_identities_verified_before_full_gzip_open": True,
    }


def validate_production_gate_g10_preflight(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a canonical production preflight report.

    This public boundary lets a preflight-only launcher prove that all source
    identities and no-analysis flags satisfy the production result contract
    without constructing a candidate result.
    """
    report = _canonical_copy(
        _mapping(value, "production Gate G10 preflight"),
        "production Gate G10 preflight",
    )
    _validate_production_preflight(report)
    return report


def _validate_candidate_result(
    value: Mapping[str, Any],
    *,
    candidate: str,
) -> dict[str, Any]:
    result = _canonical_copy(_mapping(value, f"{candidate} Gate G10 result"), candidate)
    expected_result_keys = {
        "version",
        "gate_id",
        "candidate",
        "metric",
        "unit",
        "groups",
        "completions",
        "zero_variance_groups",
        "varying_groups",
        "zero_variance_share",
        "unique_reward_levels_per_group_histogram",
        "ordered_group_id_sha256",
        "comparison",
        "locked_original_baseline",
        "boundary",
    }
    if set(result) != expected_result_keys:
        raise ValueError(f"{candidate} Gate G10 result schema drifted")
    exact_values = {
        "version": GATE_CORE_VERSION,
        "gate_id": GATE_ID,
        "candidate": candidate,
        "metric": "full authoritative-training zero-variance group share",
        "unit": "product group",
        "groups": EXPECTED_TRAINING_GROUPS,
        "completions": EXPECTED_COMPLETIONS,
        "ordered_group_id_sha256": EXPECTED_ORDERED_SKU_SHA256,
    }
    for key, expected in exact_values.items():
        if result.get(key) != expected:
            raise ValueError(f"{candidate} Gate G10 {key} drifted")
    zero = result.get("zero_variance_groups")
    varying = result.get("varying_groups")
    if not isinstance(zero, int) or isinstance(zero, bool) or not 0 <= zero <= EXPECTED_TRAINING_GROUPS:
        raise ValueError(f"{candidate} zero-variance count is invalid")
    if varying != EXPECTED_TRAINING_GROUPS - zero:
        raise ValueError(f"{candidate} varying-group count drifted")
    share = result.get("zero_variance_share")
    if not isinstance(share, (int, float)) or isinstance(share, bool):
        raise ValueError(f"{candidate} zero-variance share is invalid")
    if not math.isfinite(float(share)) or float(share) != zero / EXPECTED_TRAINING_GROUPS:
        raise ValueError(f"{candidate} zero-variance share drifted")

    histogram = _mapping(
        result.get("unique_reward_levels_per_group_histogram"),
        f"{candidate} unique-level histogram",
    )
    histogram_counts: dict[int, int] = {}
    for raw_level, count in histogram.items():
        try:
            level = int(raw_level)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{candidate} histogram level is invalid") from exc
        if str(level) != raw_level or not 1 <= level <= 8:
            raise ValueError(f"{candidate} histogram level is invalid")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"{candidate} histogram count is invalid")
        histogram_counts[level] = count
    if sum(histogram_counts.values()) != EXPECTED_TRAINING_GROUPS:
        raise ValueError(f"{candidate} histogram denominator drifted")
    if histogram_counts.get(1, 0) != zero:
        raise ValueError(f"{candidate} histogram zero-variance count drifted")

    comparison = _mapping(result.get("comparison"), f"{candidate} comparison")
    expected_pass = (
        zero * THRESHOLD_DENOMINATOR
        <= EXPECTED_TRAINING_GROUPS * THRESHOLD_NUMERATOR
    )
    expected_comparison = {
        "canonical_decimal_places": COMPARISON_DECIMALS,
        "operator": "less_than_or_equal",
        "threshold": THRESHOLD,
        "threshold_exact_fraction": "2/5",
        "maximum_allowed_zero_variance_groups": MAXIMUM_ALLOWED_ZERO_VARIANCE_GROUPS,
        "passes": expected_pass,
        "margin_groups": MAXIMUM_ALLOWED_ZERO_VARIANCE_GROUPS - zero,
        "margin_share": THRESHOLD - zero / EXPECTED_TRAINING_GROUPS,
    }
    if dict(comparison) != expected_comparison:
        raise ValueError(f"{candidate} comparison schema or value drifted")
    baseline = _mapping(
        result.get("locked_original_baseline"),
        f"{candidate} original baseline",
    )
    expected_baseline = {
        "groups": EXPECTED_TRAINING_GROUPS,
        "completions": EXPECTED_COMPLETIONS,
        "zero_variance_groups": ORIGINAL_ZERO_VARIANCE_GROUPS,
        "zero_variance_share": ORIGINAL_ZERO_VARIANCE_SHARE,
    }
    if dict(baseline) != expected_baseline:
        raise ValueError(f"{candidate} original baseline drifted")
    result_boundary = _mapping(result.get("boundary"), f"{candidate} boundary")
    expected_result_boundary = {
        "input_kind": "already-materialized in-memory candidate reward groups",
        "file_io_performed": False,
        "real_full_training_replay_opened_by_calculator": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
    }
    if dict(result_boundary) != expected_result_boundary:
        raise ValueError(f"{candidate} boundary drifted")
    return result


def _validate_collection(collection: Mapping[str, Any]) -> dict[str, Any]:
    collection = _mapping(collection, "Gate G10 collection")
    if collection.get("version") != COLLECTOR_VERSION:
        raise ValueError("Gate G10 collector version drifted")
    if collection.get("status") != "in_memory_gate_g10_completed_unverified_source_scope":
        raise ValueError("Gate G10 collector status drifted")
    if tuple(collection.get("candidate_order", ())) != EXPECTED_CANDIDATES:
        raise ValueError("Gate G10 collector candidate order drifted")
    lineage = _mapping(collection.get("lineage"), "Gate G10 collector lineage")
    expected_lineage = {
        "groups": EXPECTED_TRAINING_GROUPS,
        "completions": EXPECTED_COMPLETIONS,
        "unique_skus": EXPECTED_TRAINING_GROUPS,
        "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
        "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
        "contiguous_group_positions": True,
        "all_candidates_share_ordered_denominator": True,
        "groups_adapted_once": True,
        "calculator_calls": len(EXPECTED_CANDIDATES),
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            raise ValueError(f"Gate G10 collector lineage {key} drifted")
    per_group_hashes = lineage.get("per_group_rollout_key_sha256")
    if not isinstance(per_group_hashes, list):
        raise ValueError("Gate G10 per-group rollout hashes must be a list")
    if len(per_group_hashes) != EXPECTED_TRAINING_GROUPS:
        raise ValueError("Gate G10 per-group rollout hash denominator drifted")
    if not all(_is_sha256(value) for value in per_group_hashes):
        raise ValueError("Gate G10 per-group rollout hash is invalid")
    candidate_hashes = _mapping(
        lineage.get("candidate_ordered_group_sha256"),
        "Gate G10 candidate ordered hashes",
    )
    if set(candidate_hashes) != set(EXPECTED_CANDIDATES) or any(
        candidate_hashes.get(candidate) != EXPECTED_ORDERED_SKU_SHA256
        for candidate in EXPECTED_CANDIDATES
    ):
        raise ValueError("Gate G10 candidate ordered hashes drifted")
    boundary = _mapping(collection.get("boundary"), "Gate G10 collector boundary")
    if boundary.get("source_scope_verified_by_collector") is not False:
        raise ValueError("Gate G10 collector improperly authorized its source")
    if boundary.get("real_gate_g10_result_authorized") is not False:
        raise ValueError("Gate G10 collector improperly authorized a real result")
    _require_false(
        boundary,
        (
            "file_io_performed",
            "real_full_training_replay_opened_by_collector",
            "active_candidate_aggregates_calculated",
            "candidate_rankings_calculated",
            "winner_selected",
            "artifact_published",
        ),
        "Gate G10 collector boundary",
    )
    if boundary.get("gate_g10_calculated_for_supplied_inputs") is not True:
        raise ValueError("Gate G10 collector did not calculate supplied inputs")

    raw_results = _mapping(
        collection.get("candidate_results"),
        "Gate G10 candidate results",
    )
    if set(raw_results) != set(EXPECTED_CANDIDATES):
        raise ValueError("Gate G10 candidate result set drifted")
    return {
        candidate: _validate_candidate_result(raw_results[candidate], candidate=candidate)
        for candidate in EXPECTED_CANDIDATES
    }


def build_production_gate_g10_result(
    *,
    preflight_report: Mapping[str, Any],
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one production-only G10 artifact from validated in-memory inputs."""
    checked_preflight = validate_production_gate_g10_preflight(preflight_report)
    sources = _validate_production_preflight(checked_preflight)
    candidate_results = _validate_collection(collection)
    summary = {
        candidate: {
            "zero_variance_groups": result["zero_variance_groups"],
            "zero_variance_share": result["zero_variance_share"],
            "passes": result["comparison"]["passes"],
            "margin_groups": result["comparison"]["margin_groups"],
        }
        for candidate, result in candidate_results.items()
    }
    artifact = {
        "version": VERSION,
        "status": "production_gate_g10_completed",
        "role": ROLE,
        "candidate_order": list(EXPECTED_CANDIDATES),
        "sources": sources,
        "lineage": {
            "groups": EXPECTED_TRAINING_GROUPS,
            "completions": EXPECTED_COMPLETIONS,
            "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
            "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
            "all_candidates_share_ordered_denominator": True,
            "manifest_lineage_matches_stream": True,
        },
        "gate_contract": {
            "gate_id": GATE_ID,
            "metric": "full authoritative-training zero-variance group share",
            "operator": "less_than_or_equal",
            "threshold": THRESHOLD,
            "threshold_exact_fraction": "2/5",
            "maximum_allowed_zero_variance_groups": (
                MAXIMUM_ALLOWED_ZERO_VARIANCE_GROUPS
            ),
            "locked_original_zero_variance_groups": (
                ORIGINAL_ZERO_VARIANCE_GROUPS
            ),
            "locked_original_zero_variance_share": ORIGINAL_ZERO_VARIANCE_SHARE,
        },
        "candidate_results": candidate_results,
        "gate_summary": summary,
        "selection_boundary": {
            "real_full_training_replay_used": True,
            "gate_g10_calculated": True,
            "gate_g10_threshold_applied": True,
            "active_candidate_aggregates_calculated": False,
            "gates_g1_through_g9_applied": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
            "gpu_training_authorized": False,
        },
        "interpretation_guardrails": {
            "candidate_superiority_claim_allowed": False,
            "gate_g10_alone_can_select_candidate": False,
            "next_step": (
                "analyze the active replay under locked D3 metrics and Gates "
                "G1-G9 before any candidate selection"
            ),
        },
    }
    validate_production_gate_g10_result(artifact)
    return artifact


def validate_production_gate_g10_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete durable schema before any publication attempt."""
    artifact = _canonical_copy(_mapping(value, "Gate G10 artifact"), "artifact")
    if artifact.get("version") != VERSION or artifact.get("role") != ROLE:
        raise ValueError("Gate G10 artifact identity drifted")
    if artifact.get("status") != "production_gate_g10_completed":
        raise ValueError("Gate G10 artifact status drifted")
    expected_top_level_keys = {
        "version",
        "status",
        "role",
        "candidate_order",
        "sources",
        "lineage",
        "gate_contract",
        "candidate_results",
        "gate_summary",
        "selection_boundary",
        "interpretation_guardrails",
    }
    if set(artifact) != expected_top_level_keys:
        raise ValueError("Gate G10 artifact top-level schema drifted")
    if tuple(artifact.get("candidate_order", ())) != EXPECTED_CANDIDATES:
        raise ValueError("Gate G10 artifact candidate order drifted")
    lineage = _mapping(artifact.get("lineage"), "Gate G10 artifact lineage")
    expected_lineage = {
        "groups": EXPECTED_TRAINING_GROUPS,
        "completions": EXPECTED_COMPLETIONS,
        "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
        "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
        "all_candidates_share_ordered_denominator": True,
        "manifest_lineage_matches_stream": True,
    }
    if dict(lineage) != expected_lineage:
        raise ValueError("Gate G10 artifact lineage drifted")
    gate_contract = _mapping(
        artifact.get("gate_contract"),
        "Gate G10 artifact gate contract",
    )
    expected_gate_contract = {
        "gate_id": GATE_ID,
        "metric": "full authoritative-training zero-variance group share",
        "operator": "less_than_or_equal",
        "threshold": THRESHOLD,
        "threshold_exact_fraction": "2/5",
        "maximum_allowed_zero_variance_groups": (
            MAXIMUM_ALLOWED_ZERO_VARIANCE_GROUPS
        ),
        "locked_original_zero_variance_groups": ORIGINAL_ZERO_VARIANCE_GROUPS,
        "locked_original_zero_variance_share": ORIGINAL_ZERO_VARIANCE_SHARE,
    }
    if dict(gate_contract) != expected_gate_contract:
        raise ValueError("Gate G10 artifact gate contract drifted")
    candidate_results = _mapping(
        artifact.get("candidate_results"),
        "Gate G10 artifact candidate results",
    )
    if set(candidate_results) != set(EXPECTED_CANDIDATES):
        raise ValueError("Gate G10 artifact candidate set drifted")
    checked_results = {
        candidate: _validate_candidate_result(
            candidate_results[candidate],
            candidate=candidate,
        )
        for candidate in EXPECTED_CANDIDATES
    }
    expected_summary = {
        candidate: {
            "zero_variance_groups": result["zero_variance_groups"],
            "zero_variance_share": result["zero_variance_share"],
            "passes": result["comparison"]["passes"],
            "margin_groups": result["comparison"]["margin_groups"],
        }
        for candidate, result in checked_results.items()
    }
    if artifact.get("gate_summary") != expected_summary:
        raise ValueError("Gate G10 artifact summary drifted")
    selection = _mapping(
        artifact.get("selection_boundary"),
        "Gate G10 artifact selection boundary",
    )
    expected_selection = {
        "real_full_training_replay_used": True,
        "gate_g10_calculated": True,
        "gate_g10_threshold_applied": True,
        "active_candidate_aggregates_calculated": False,
        "gates_g1_through_g9_applied": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
        "gpu_training_authorized": False,
    }
    if dict(selection) != expected_selection:
        raise ValueError("Gate G10 artifact selection boundary drifted")
    guardrails = _mapping(
        artifact.get("interpretation_guardrails"),
        "Gate G10 artifact interpretation guardrails",
    )
    expected_guardrails = {
        "candidate_superiority_claim_allowed": False,
        "gate_g10_alone_can_select_candidate": False,
        "next_step": (
            "analyze the active replay under locked D3 metrics and Gates G1-G9 "
            "before any candidate selection"
        ),
    }
    if dict(guardrails) != expected_guardrails:
        raise ValueError("Gate G10 artifact interpretation guardrails drifted")
    sources = _mapping(artifact.get("sources"), "Gate G10 artifact sources")
    expected_source_keys = {
        "preflight_version",
        "active_manifest",
        "active_records",
        "comparison_contract",
        "class_weights",
        "full_manifest",
        "full_records",
        "cb_extension_ledger_sha256",
        "all_identities_verified_before_full_gzip_open",
    }
    if set(sources) != expected_source_keys:
        raise ValueError("Gate G10 artifact source schema drifted")
    if sources.get("all_identities_verified_before_full_gzip_open") is not True:
        raise ValueError("Gate G10 artifact source authorization drifted")
    _validate_production_preflight(
        {
            "version": sources["preflight_version"],
            "status": "production_preflight_passed",
            "mode": "locked_dual_replay",
            "role": PRODUCTION_ROLE,
            "comparison_contract_version": COMPARISON_CONTRACT_VERSION,
            "inputs": {
                "all_identities_verified_before_gzip_open": True,
                "manifest": sources["active_manifest"],
                "records": sources["active_records"],
                "comparison_contract": sources["comparison_contract"],
                "class_weights": sources["class_weights"],
            },
            "full_training_replay": {
                "role": FULL_TRAINING_ROLE,
                "groups": EXPECTED_TRAINING_GROUPS,
                "completions": EXPECTED_COMPLETIONS,
                "active_groups_included": 1_438,
                "additional_training_groups": 1_802,
                "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
                "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
                "scope_is_distinct_from_active_replay": True,
                "all_identities_verified_before_gzip_open": True,
                "manifest": sources["full_manifest"],
                "records": sources["full_records"],
                "cb_extension": {
                    "ordered_entry_ledger_sha256": sources[
                        "cb_extension_ledger_sha256"
                    ],
                    "ledger_hash_recomputed": True,
                    "active_weights_changed": False,
                    "candidate_aggregates_calculated": False,
                },
            },
            "settings": {"settings_locked_to_production_contract": True},
            "selection_boundary": {
                "replay_gzip_decompressed": False,
                "replay_records_parsed": False,
                "full_replay_gzip_decompressed": False,
                "full_replay_records_parsed": False,
                "candidate_aggregate_metrics_calculated": False,
                "gate_g10_calculated": False,
                "acceptance_gates_applied": False,
                "candidate_rankings_calculated": False,
                "winner_selected": False,
                "artifact_published": False,
            },
        }
    )
    return artifact


def publish_production_gate_g10_result(
    *,
    repo_root: str | Path,
    artifact: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate and publish exclusively to the single locked result path."""
    root = Path(repo_root).resolve()
    locked = (root / DEFAULT_OUTPUT).resolve()
    requested = Path(output_path)
    requested = (
        requested.resolve()
        if requested.is_absolute()
        else (root / requested).resolve()
    )
    if requested != locked:
        raise ValueError(f"Gate G10 output path must be exactly {DEFAULT_OUTPUT}")
    checked = validate_production_gate_g10_result(artifact)
    write_exclusive_atomic_json(locked, checked)
    return {
        "path": DEFAULT_OUTPUT,
        "bytes": locked.stat().st_size,
        "sha256": sha256_file(locked),
        "published_exclusively": True,
        "artifact": checked,
    }
