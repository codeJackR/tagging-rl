from __future__ import annotations

import json
from copy import deepcopy

import pytest

from training.audit_data_boundaries import sha256_file
from training.run2_analysis_orchestrator import VERSION as PREFLIGHT_VERSION
from training.run2_comparison_contract import VERSION as COMPARISON_VERSION
from training.run2_gate_g10 import (
    ORIGINAL_ZERO_VARIANCE_GROUPS,
    ORIGINAL_ZERO_VARIANCE_SHARE,
    VERSION as CORE_VERSION,
)
from training.run2_gate_g10_collector import VERSION as COLLECTOR_VERSION
from training.run2_gate_g10_result_contract import (
    DEFAULT_OUTPUT,
    EXPECTED_ACTIVE_MANIFEST_SHA256,
    EXPECTED_ACTIVE_RECORDS_SHA256,
    EXPECTED_CB_EXTENSION_LEDGER_SHA256,
    EXPECTED_CLASS_WEIGHTS_SHA256,
    EXPECTED_COMPARISON_CONTRACT_SHA256,
    EXPECTED_FULL_MANIFEST_SHA256,
    EXPECTED_FULL_RECORDS_SHA256,
    EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
    EXPECTED_ORDERED_SKU_SHA256,
    MAXIMUM_ALLOWED_ZERO_VARIANCE_GROUPS,
    ROLE,
    VERSION,
    build_production_gate_g10_result,
    publish_production_gate_g10_result,
    validate_production_gate_g10_result,
)


CANDIDATES = ("U", "UA", "CB")
GROUPS = 3_240
COMPLETIONS = 25_920


def _metadata(path, size, sha256):
    return {"path": path, "bytes": size, "sha256": sha256}


def _preflight():
    return {
        "version": PREFLIGHT_VERSION,
        "status": "production_preflight_passed",
        "mode": "locked_dual_replay",
        "role": "training_only_identical_group_candidate_replay_records",
        "comparison_contract_version": COMPARISON_VERSION,
        "inputs": {
            "all_identities_verified_before_gzip_open": True,
            "manifest": _metadata(
                "runs/grpo-run2-candidate-replay-manifest.json",
                6_435,
                EXPECTED_ACTIVE_MANIFEST_SHA256,
            ),
            "records": _metadata(
                "runs/grpo-run2-candidate-replay-records.jsonl.gz",
                1_921_202,
                EXPECTED_ACTIVE_RECORDS_SHA256,
            ),
            "comparison_contract": _metadata(
                "runs/grpo-run2-comparison-contract.json",
                12_079,
                EXPECTED_COMPARISON_CONTRACT_SHA256,
            ),
            "class_weights": _metadata(
                "runs/grpo-run2-cb-class-weights.json",
                27_446,
                EXPECTED_CLASS_WEIGHTS_SHA256,
            ),
        },
        "full_training_replay": {
            "role": "full_authoritative_training_identical_group_candidate_replay_records",
            "groups": GROUPS,
            "completions": COMPLETIONS,
            "active_groups_included": 1_438,
            "additional_training_groups": 1_802,
            "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
            "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
            "scope_is_distinct_from_active_replay": True,
            "all_identities_verified_before_gzip_open": True,
            "manifest": _metadata(
                "runs/grpo-run2-full-training-candidate-replay-manifest.json",
                10_709,
                EXPECTED_FULL_MANIFEST_SHA256,
            ),
            "records": _metadata(
                "runs/grpo-run2-full-training-candidate-replay-records.jsonl.gz",
                4_168_170,
                EXPECTED_FULL_RECORDS_SHA256,
            ),
            "cb_extension": {
                "ordered_entry_ledger_sha256": EXPECTED_CB_EXTENSION_LEDGER_SHA256,
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


def _candidate_result(candidate, zero):
    share = zero / GROUPS
    passes = zero <= MAXIMUM_ALLOWED_ZERO_VARIANCE_GROUPS
    histogram = {"1": zero, "2": GROUPS - zero}
    if zero == 0:
        histogram = {"2": GROUPS}
    if zero == GROUPS:
        histogram = {"1": GROUPS}
    return {
        "version": CORE_VERSION,
        "gate_id": "G10_full_training_variation",
        "candidate": candidate,
        "metric": "full authoritative-training zero-variance group share",
        "unit": "product group",
        "groups": GROUPS,
        "completions": COMPLETIONS,
        "zero_variance_groups": zero,
        "varying_groups": GROUPS - zero,
        "zero_variance_share": share,
        "unique_reward_levels_per_group_histogram": histogram,
        "ordered_group_id_sha256": EXPECTED_ORDERED_SKU_SHA256,
        "comparison": {
            "canonical_decimal_places": 12,
            "operator": "less_than_or_equal",
            "threshold": 0.4,
            "threshold_exact_fraction": "2/5",
            "maximum_allowed_zero_variance_groups": 1_296,
            "passes": passes,
            "margin_groups": 1_296 - zero,
            "margin_share": 0.4 - share,
        },
        "locked_original_baseline": {
            "groups": GROUPS,
            "completions": COMPLETIONS,
            "zero_variance_groups": ORIGINAL_ZERO_VARIANCE_GROUPS,
            "zero_variance_share": ORIGINAL_ZERO_VARIANCE_SHARE,
        },
        "boundary": {
            "input_kind": "already-materialized in-memory candidate reward groups",
            "file_io_performed": False,
            "real_full_training_replay_opened_by_calculator": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
        },
    }


def _collection():
    results = {
        "U": _candidate_result("U", 1_000),
        "UA": _candidate_result("UA", 1_296),
        "CB": _candidate_result("CB", 1_400),
    }
    return {
        "version": COLLECTOR_VERSION,
        "status": "in_memory_gate_g10_completed_unverified_source_scope",
        "candidate_order": list(CANDIDATES),
        "lineage": {
            "groups": GROUPS,
            "completions": COMPLETIONS,
            "unique_skus": GROUPS,
            "ordered_sku_sha256": EXPECTED_ORDERED_SKU_SHA256,
            "ordered_rollout_key_sha256": EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
            "per_group_rollout_key_sha256": ["0" * 64] * GROUPS,
            "candidate_ordered_group_sha256": {
                candidate: EXPECTED_ORDERED_SKU_SHA256 for candidate in CANDIDATES
            },
            "contiguous_group_positions": True,
            "all_candidates_share_ordered_denominator": True,
            "groups_adapted_once": True,
            "calculator_calls": 3,
        },
        "candidate_results": results,
        "boundary": {
            "input_kind": "ordered in-memory nested replay groups",
            "file_io_performed": False,
            "real_full_training_replay_opened_by_collector": False,
            "gate_g10_calculated_for_supplied_inputs": True,
            "source_scope_verified_by_collector": False,
            "real_gate_g10_result_authorized": False,
            "active_candidate_aggregates_calculated": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
            "artifact_published": False,
        },
    }


def _artifact():
    return build_production_gate_g10_result(
        preflight_report=_preflight(),
        collection=_collection(),
    )


def test_build_locks_sources_lineage_results_and_selection_boundary():
    artifact = _artifact()

    assert artifact["version"] == VERSION
    assert artifact["status"] == "production_gate_g10_completed"
    assert artifact["role"] == ROLE
    assert artifact["candidate_order"] == ["U", "UA", "CB"]
    assert artifact["lineage"]["groups"] == GROUPS
    assert artifact["sources"]["full_records"]["sha256"] == (
        EXPECTED_FULL_RECORDS_SHA256
    )
    assert artifact["gate_summary"] == {
        "U": {
            "zero_variance_groups": 1_000,
            "zero_variance_share": 1_000 / GROUPS,
            "passes": True,
            "margin_groups": 296,
        },
        "UA": {
            "zero_variance_groups": 1_296,
            "zero_variance_share": 0.4,
            "passes": True,
            "margin_groups": 0,
        },
        "CB": {
            "zero_variance_groups": 1_400,
            "zero_variance_share": 1_400 / GROUPS,
            "passes": False,
            "margin_groups": -104,
        },
    }
    assert artifact["selection_boundary"]["gate_g10_calculated"] is True
    assert artifact["selection_boundary"]["gates_g1_through_g9_applied"] is False
    assert artifact["selection_boundary"]["winner_selected"] is False
    assert validate_production_gate_g10_result(artifact) == artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["selection_boundary"].update(
                full_replay_gzip_decompressed=True
            ),
            "full_replay_gzip_decompressed",
        ),
        (
            lambda value: value["full_training_replay"]["records"].update(
                sha256="0" * 64
            ),
            "records SHA-256",
        ),
        (
            lambda value: value["full_training_replay"]["cb_extension"].update(
                active_weights_changed=True
            ),
            "changed active weights",
        ),
    ],
)
def test_preflight_drift_fails_before_result_build(mutate, message):
    preflight = _preflight()
    mutate(preflight)
    with pytest.raises(ValueError, match=message):
        build_production_gate_g10_result(
            preflight_report=preflight,
            collection=_collection(),
        )


def test_collection_lineage_and_candidate_arithmetic_drift_fail_closed():
    bad_lineage = _collection()
    bad_lineage["lineage"]["ordered_sku_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ordered_sku_sha256"):
        build_production_gate_g10_result(
            preflight_report=_preflight(),
            collection=bad_lineage,
        )

    bad_per_group_hashes = _collection()
    bad_per_group_hashes["lineage"]["per_group_rollout_key_sha256"].pop()
    with pytest.raises(ValueError, match="rollout hash denominator"):
        build_production_gate_g10_result(
            preflight_report=_preflight(),
            collection=bad_per_group_hashes,
        )

    bad_share = _collection()
    bad_share["candidate_results"]["U"]["zero_variance_share"] = 0.0
    with pytest.raises(ValueError, match="zero-variance share drifted"):
        build_production_gate_g10_result(
            preflight_report=_preflight(),
            collection=bad_share,
        )

    bad_pass = _collection()
    bad_pass["candidate_results"]["CB"]["comparison"]["passes"] = True
    with pytest.raises(ValueError, match="comparison schema or value drifted"):
        build_production_gate_g10_result(
            preflight_report=_preflight(),
            collection=bad_pass,
        )


def test_locked_path_publication_is_deterministic_and_exclusive(tmp_path):
    artifact = _artifact()
    published = publish_production_gate_g10_result(
        repo_root=tmp_path,
        artifact=artifact,
    )
    output = tmp_path / DEFAULT_OUTPUT

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert published["path"] == DEFAULT_OUTPUT
    assert published["bytes"] == output.stat().st_size
    assert published["sha256"] == sha256_file(output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="output already exists"):
        publish_production_gate_g10_result(repo_root=tmp_path, artifact=artifact)
    assert output.read_bytes() == original


def test_nonlocked_path_and_tampered_artifact_are_never_published(tmp_path):
    artifact = _artifact()
    with pytest.raises(ValueError, match="must be exactly"):
        publish_production_gate_g10_result(
            repo_root=tmp_path,
            artifact=artifact,
            output_path="runs/other.json",
        )
    tampered = deepcopy(artifact)
    tampered["gate_summary"]["U"]["passes"] = False
    with pytest.raises(ValueError, match="summary drifted"):
        publish_production_gate_g10_result(
            repo_root=tmp_path,
            artifact=tampered,
        )
    assert not (tmp_path / DEFAULT_OUTPUT).exists()


def test_gate_contract_and_schema_drift_are_never_published(tmp_path):
    bad_threshold = _artifact()
    bad_threshold["gate_contract"]["threshold"] = 0.5
    with pytest.raises(ValueError, match="gate contract drifted"):
        publish_production_gate_g10_result(
            repo_root=tmp_path,
            artifact=bad_threshold,
        )

    extra_field = _artifact()
    extra_field["selection_decision"] = "U"
    with pytest.raises(ValueError, match="top-level schema drifted"):
        publish_production_gate_g10_result(
            repo_root=tmp_path,
            artifact=extra_field,
        )

    nested_decision = _artifact()
    nested_decision["candidate_results"]["U"]["selected"] = True
    with pytest.raises(ValueError, match="result schema drifted"):
        publish_production_gate_g10_result(
            repo_root=tmp_path,
            artifact=nested_decision,
        )
    assert not (tmp_path / DEFAULT_OUTPUT).exists()


def test_atomic_link_failure_leaves_no_result_or_temporary_file(tmp_path, monkeypatch):
    from training import audit_data_boundaries

    def fail_link(*args, **kwargs):
        raise OSError("synthetic link failure")

    monkeypatch.setattr(audit_data_boundaries.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic link failure"):
        publish_production_gate_g10_result(
            repo_root=tmp_path,
            artifact=_artifact(),
        )
    output = tmp_path / DEFAULT_OUTPUT
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
