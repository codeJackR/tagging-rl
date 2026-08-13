from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from training.analyze_run2_candidates import REWARD_LABELS, VERSION as CORE_VERSION
from training.audit_data_boundaries import sha256_file
from training.run2_analysis_orchestrator import PRODUCTION_ROLE, VERSION as ORCHESTRATOR_VERSION
from training.run2_comparison_contract import VERSION as COMPARISON_VERSION
from training.run2_d3_result_contract import (
    DEFAULT_OUTPUT,
    EXPECTED_ACTIVE_MANIFEST,
    EXPECTED_ACTIVE_RECORDS,
    EXPECTED_CLASS_WEIGHTS,
    EXPECTED_COMPARISON_CONTRACT,
    EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
    EXPECTED_ORDERED_SKU_SHA256,
    EXPECTED_PAIRED_METRICS,
    EXPECTED_SEGMENT_DIMENSIONS,
    EXPECTED_TARGETS,
    publish_production_d3_result,
    validate_production_d3_preflight,
    validate_production_d3_result,
)
from training.run2_segment_summaries import VERSION as DIAGNOSTICS_VERSION


ROOT = Path(__file__).resolve().parents[1]
GROUPS = 1_438
COMPLETIONS = 11_504


def _ordered_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _group_ids():
    values = []
    path = ROOT / "data/train_weak_grpo_cap4_sft_train_v1.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        values.append(json.loads(line)["sku_id"])
    assert len(values) == GROUPS
    assert _ordered_hash(values) == EXPECTED_ORDERED_SKU_SHA256
    return values


def _distribution(count, value=0.0):
    return {
        "count": count,
        "minimum": value,
        "p05": value,
        "p25": value,
        "median": value,
        "mean": value,
        "p75": value,
        "p95": value,
        "maximum": value,
        "population_std": 0.0,
        "histogram": {str(value): count},
    }


def _reward_summary(group_ids):
    return {
        "groups": GROUPS,
        "completions": COMPLETIONS,
        "completion_reward_distribution": _distribution(COMPLETIONS),
        "group_mean_distribution": _distribution(GROUPS),
        "within_group_variance_own_scale_distribution": _distribution(GROUPS),
        "pairwise_discrimination_distribution": _distribution(GROUPS, 0.5),
        "unique_reward_levels_per_group_histogram": {"2": GROUPS},
        "largest_tie_size_per_group_histogram": {"4": GROUPS},
        "zero_variance_groups": 0,
        "zero_variance_share": 0.0,
        "groups_with_at_least_three_levels": 0,
        "groups_with_at_least_three_levels_share": 0.0,
        "groups_with_largest_tie_at_least_six": 0,
        "groups_with_largest_tie_at_least_six_share": 0.0,
        "group_values_for_paired_analysis": {
            group_id: {"pairwise_discrimination_rate": 0.5}
            for group_id in group_ids
        },
    }


def _directional(group_ids, *, contributes):
    if not contributes:
        return {
            "groups_total": GROUPS,
            "groups_contributing": 0,
            "comparable_pairs": 0,
            "concordant_pairs": 0,
            "discordant_pairs": 0,
            "reward_ties": 0,
            "group_net_alignment_distribution": None,
            "group_values_for_paired_analysis": {},
        }
    return {
        "groups_total": GROUPS,
        "groups_contributing": GROUPS,
        "comparable_pairs": GROUPS,
        "concordant_pairs": GROUPS,
        "discordant_pairs": 0,
        "reward_ties": 0,
        "group_net_alignment_distribution": _distribution(GROUPS, 1.0),
        "group_values_for_paired_analysis": {
            group_id: 1.0 for group_id in group_ids
        },
    }


def _harmful(group_ids):
    return {
        "groups_total": GROUPS,
        "groups_contributing": GROUPS,
        "comparable_pairs": GROUPS,
        "harmful_preferences": 0,
        "safe_preferences": GROUPS,
        "reward_ties": 0,
        "group_score_distribution": _distribution(GROUPS, 0.0),
        "group_values_for_paired_analysis": {
            group_id: 0.0 for group_id in group_ids
        },
    }


def _point_summary(value):
    return {"point": value, "mean": value, "median": value, "ci": [value, value]}


def _bootstrap():
    return {
        "method": "paired nonparametric product-group bootstrap, percentile interval",
        "pairing_unit": "entire k=8 product group",
        "completion_level_resampling": False,
        "groups_per_replicate": GROUPS,
        "seed": 20_260_812,
        "replicates": 10_000,
        "confidence": 0.95,
        "percentile_method": "linear interpolation at sorted index (n - 1) * p",
        "replicate_stream_sha256": "0" * 64,
        "baseline": _point_summary(0.0),
        "candidate": _point_summary(0.1),
        "delta_candidate_minus_baseline": {
            **_point_summary(0.1),
            "fraction_below_zero": 0.0,
            "fraction_equal_zero": 0.0,
            "fraction_above_zero": 1.0,
        },
    }


def _segment(group_ids):
    return {
        "groups": GROUPS,
        "completions": COMPLETIONS,
        "ordered_group_ids": list(group_ids),
        "ordered_group_id_sha256": EXPECTED_ORDERED_SKU_SHA256,
        "interpretation_allowed": True,
        "minimum_groups_for_interpretation": 30,
        "reward_summaries": {label: {} for label in REWARD_LABELS},
        "directional_alignments": {label: {} for label in REWARD_LABELS},
        "harmful_coverage": {label: {} for label in REWARD_LABELS},
    }


def _artifact():
    group_ids = _group_ids()
    reward_summaries = {
        label: _reward_summary(group_ids) for label in REWARD_LABELS
    }


    directional = {
        label: {
            target: _directional(
                group_ids,
                contributes=target == "canonical_known_utility",
            )
            for target in EXPECTED_TARGETS
        }
        for label in REWARD_LABELS
    }
    harmful = {label: _harmful(group_ids) for label in REWARD_LABELS}
    paired = {
        candidate: {metric: _bootstrap() for metric in EXPECTED_PAIRED_METRICS}
        for candidate in ("U", "UA", "CB")
    }
    field_contributions = {
        candidate: {
            "dominance_threshold_reference": 0.2,
            "dominance_gate_applied": False,
        }
        for candidate in ("U", "UA", "CB")
    }
    dimensions = {
        dimension: {
            "segments": {"fixture_all": _segment(group_ids)},
            "segment_count": 1,
            "group_memberships": GROUPS,
            "every_group_appears_exactly_once": True,
        }
        for dimension in EXPECTED_SEGMENT_DIMENSIONS
    }
    return {
        "version": ORCHESTRATOR_VERSION,
        "status": "aggregate_candidate_analysis_completed_pending_gates",
        "role": PRODUCTION_ROLE,
        "mode": "locked_active_replay",
        "selection_boundary": {
            "candidate_aggregate_metrics_calculated": True,
            "real_candidate_replay_used": True,
            "acceptance_gates_applied": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
        },
        "inputs": {
            "manifest": dict(EXPECTED_ACTIVE_MANIFEST),
            "records": dict(EXPECTED_ACTIVE_RECORDS),
            "comparison_contract": dict(EXPECTED_COMPARISON_CONTRACT),
            "class_weights": dict(EXPECTED_CLASS_WEIGHTS),
            "all_identities_verified_before_gzip_open": True,
        },
        "lineage": {
            "groups": GROUPS,
            "completions": COMPLETIONS,
            "unique_skus": GROUPS,
            "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
            "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
            "groups_streamed_once": True,
            "groups_adapted_once": True,
        },
        "settings": {
            "bootstrap_seed": 20_260_812,
            "bootstrap_replicates": 10_000,
            "confidence": 0.95,
            "settings_locked_to_production_contract": True,
        },
        "implementation": {
            "path": "training/run2_analysis_orchestrator.py",
            "bytes": 1,
            "sha256": "0" * 64,
        },
        "comparison_contract_version": COMPARISON_VERSION,
        "analysis_core": {
            "version": CORE_VERSION,
            "status": "analysis_core_completed",
            "boundary": {
                "input_kind": "already-materialized in-memory observations",
                "file_io_performed": False,
                "real_candidate_replay_opened": False,
                "acceptance_gates_applied": False,
                "winner_selected": False,
            },
            "groups": GROUPS,
            "completions": COMPLETIONS,
            "group_order": list(group_ids),
            "reward_summaries": reward_summaries,
            "directional_alignments": directional,
            "harmful_coverage": harmful,
            "paired_candidate_minus_original": paired,
        },
        "contribution_and_segment_diagnostics": {
            "version": DIAGNOSTICS_VERSION,
            "status": "in_memory_diagnostics_completed",
            "boundary": {
                "file_io_performed": False,
                "real_candidate_replay_opened_by_this_module": False,
                "acceptance_gates_applied": False,
                "winner_selected": False,
            },
            "dominance": {
                "version": DIAGNOSTICS_VERSION,
                "groups": GROUPS,
                "field_contributions": field_contributions,
                "cb_class_contributions": {
                    "dominance_threshold_reference": 0.15,
                    "dominance_gate_applied": False,
                },
                "grain_guardrails": {
                    "product_counts_derived_from_group_observations_only": True,
                    "field_child_rows_used_as_product_denominator": False,
                    "class_child_rows_used_as_product_denominator": False,
                    "class_allocations_reconstruct_cb_known_fields": True,
                },
            },
            "product_segments": {
                "version": DIAGNOSTICS_VERSION,
                "groups": GROUPS,
                "dimensions": dimensions,
                "segment_keys_are_pre_outcome": True,
                "bootstrap_performed": False,
                "acceptance_gates_applied": False,
                "winner_selected": False,
            },
        },
    }


def _preflight():
    return {
        "version": ORCHESTRATOR_VERSION,
        "status": "production_active_preflight_passed",
        "mode": "locked_active_replay",
        "role": PRODUCTION_ROLE,
        "inputs": {
            "manifest": dict(EXPECTED_ACTIVE_MANIFEST),
            "records": dict(EXPECTED_ACTIVE_RECORDS),
            "comparison_contract": dict(EXPECTED_COMPARISON_CONTRACT),
            "class_weights": dict(EXPECTED_CLASS_WEIGHTS),
            "all_identities_verified_before_gzip_open": True,
        },
        "lineage": {
            "groups": GROUPS,
            "completions": COMPLETIONS,
            "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
            "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
            "contract_lineage_matches_manifest": True,
        },
        "settings": {
            "bootstrap_seed": 20_260_812,
            "bootstrap_replicates": 10_000,
            "confidence": 0.95,
            "settings_locked_to_production_contract": True,
        },
        "comparison_contract_version": COMPARISON_VERSION,
        "selection_boundary": {
            "replay_gzip_decompressed": False,
            "replay_records_parsed": False,
            "candidate_aggregate_metrics_calculated": False,
            "acceptance_gates_applied": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
            "artifact_published": False,
        },
    }


def test_valid_production_preflight_passes_and_drift_fails_closed():
    report = _preflight()
    assert validate_production_d3_preflight(report) == report

    drifted = _preflight()
    drifted["selection_boundary"]["replay_records_parsed"] = True
    with pytest.raises(ValueError, match="selection boundary drifted"):
        validate_production_d3_preflight(drifted)

    wrong_source = _preflight()
    wrong_source["inputs"]["records"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="input identities drifted"):
        validate_production_d3_preflight(wrong_source)


def test_valid_production_shaped_artifact_passes_without_replay_io():
    artifact = _artifact()
    assert validate_production_d3_result(artifact) == artifact


def test_source_lineage_and_settings_drift_fail_closed():
    source = _artifact()
    source["inputs"]["records"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="input identities drifted"):
        validate_production_d3_result(source)

    lineage = _artifact()
    lineage["lineage"]["groups"] -= 1
    with pytest.raises(ValueError, match="lineage drifted"):
        validate_production_d3_result(lineage)

    settings = _artifact()
    settings["settings"]["bootstrap_replicates"] = 9_999
    with pytest.raises(ValueError, match="settings drifted"):
        validate_production_d3_result(settings)


def test_reward_denominator_histogram_and_group_order_drift_fail_closed():
    histogram = _artifact()
    histogram["analysis_core"]["reward_summaries"]["U"][
        "unique_reward_levels_per_group_histogram"
    ]["2"] -= 1
    with pytest.raises(ValueError, match="unique-level denominator drifted"):
        validate_production_d3_result(histogram)

    duplicate = _artifact()
    duplicate["analysis_core"]["group_order"][1] = duplicate["analysis_core"][
        "group_order"
    ][0]
    with pytest.raises(ValueError, match="not unique"):
        validate_production_d3_result(duplicate)


def test_bootstrap_and_selection_drift_fail_closed():
    bootstrap = _artifact()
    bootstrap["analysis_core"]["paired_candidate_minus_original"]["UA"][
        "pairwise_discrimination"
    ]["replicates"] = 100
    with pytest.raises(ValueError, match="replicate count drifted"):
        validate_production_d3_result(bootstrap)

    selected = _artifact()
    selected["selection_boundary"]["winner_selected"] = True
    with pytest.raises(ValueError, match="selection boundary drifted"):
        validate_production_d3_result(selected)


def test_segment_partition_and_dominance_gate_drift_fail_closed():
    segment = _artifact()
    member = segment["contribution_and_segment_diagnostics"]["product_segments"][
        "dimensions"
    ]["product_category"]["segments"]["fixture_all"]
    member["ordered_group_ids"].pop()
    with pytest.raises(ValueError, match="denominator drifted"):
        validate_production_d3_result(segment)

    dominance = _artifact()
    dominance["contribution_and_segment_diagnostics"]["dominance"][
        "field_contributions"
    ]["CB"]["dominance_gate_applied"] = True
    with pytest.raises(ValueError, match="field gate was applied"):
        validate_production_d3_result(dominance)


def test_locked_publication_is_deterministic_exclusive_and_path_restricted(tmp_path):
    artifact = _artifact()
    published = publish_production_d3_result(repo_root=tmp_path, artifact=artifact)
    output = tmp_path / DEFAULT_OUTPUT
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert published["path"] == DEFAULT_OUTPUT
    assert published["bytes"] == output.stat().st_size
    assert published["sha256"] == sha256_file(output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="output already exists"):
        publish_production_d3_result(repo_root=tmp_path, artifact=artifact)
    assert output.read_bytes() == original

    other_root = tmp_path / "other"
    with pytest.raises(ValueError, match="must be exactly"):
        publish_production_d3_result(
            repo_root=other_root,
            artifact=artifact,
            output_path="runs/not-d3.json",
        )


def test_atomic_failure_leaves_no_result_or_temporary_file(tmp_path, monkeypatch):
    from training import audit_data_boundaries

    def fail_link(*args, **kwargs):
        raise OSError("synthetic D3 link failure")

    monkeypatch.setattr(audit_data_boundaries.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic D3 link failure"):
        publish_production_d3_result(repo_root=tmp_path, artifact=_artifact())
    output = tmp_path / DEFAULT_OUTPUT
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_nonfinite_or_unknown_schema_never_publishes(tmp_path):
    nonfinite = _artifact()
    nonfinite["analysis_core"]["reward_summaries"]["CB"][
        "zero_variance_share"
    ] = float("nan")
    with pytest.raises(ValueError, match="finite canonical JSON"):
        publish_production_d3_result(repo_root=tmp_path, artifact=nonfinite)

    unknown = _artifact()
    unknown["winner"] = "CB"
    with pytest.raises(ValueError, match="artifact schema drifted"):
        publish_production_d3_result(repo_root=tmp_path, artifact=unknown)
    assert not (tmp_path / DEFAULT_OUTPUT).exists()
