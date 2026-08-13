from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_run2_replay_adapter import _class_artifact, _group
from training.audit_data_boundaries import sha256_file
from training.replay_run2_candidates import write_replay_groups
from training.run2_analysis_orchestrator import (
    EXPECTED_FULL_MANIFEST_BYTES,
    EXPECTED_FULL_MANIFEST_SHA256,
    EXPECTED_FULL_RECORDS_BYTES,
    EXPECTED_FULL_RECORDS_SHA256,
    FULL_SYNTHETIC_ROLE,
    SYNTHETIC_ROLE,
    VERSION,
    build_analysis_artifact,
    parse_args,
    run_active_preflight,
    run_analysis,
    run_preflight,
)
from training.replay_run2_full_training_candidates import (
    CB_DIAGNOSTIC_EXTENSION_VERSION,
    MANIFEST_VERSION as FULL_REPLAY_VERSION,
)
from training.run2_comparison_contract import VERSION as CONTRACT_VERSION
from training.replay_run2_candidates import VERSION as REPLAY_VERSION


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture(tmp_path: Path, *, groups_count: int = 2):
    groups = []
    for position in range(groups_count):
        group = deepcopy(_group())
        sku_id = f"synthetic-sku-{position}"
        group["group_position"] = position
        group["sku_id"] = sku_id
        for completion in group["completions"]:
            completion["source_rollout"]["sku_id"] = sku_id
        groups.append(group)

    records = tmp_path / "synthetic-records.jsonl.gz"
    records_meta = write_replay_groups(records, groups, repo_root=tmp_path)
    class_weights = tmp_path / "synthetic-class-weights.json"
    _write_json(class_weights, _class_artifact())
    manifest = {
        "version": REPLAY_VERSION,
        "status": "raw_evidence_published",
        "role": SYNTHETIC_ROLE,
        "selection_boundary": {
            "aggregate_candidate_comparison_calculated": False,
            "candidate_rankings_calculated": False,
            "acceptance_thresholds_applied": False,
            "winner_selected": False,
        },
        "record_contract": {
            "group_size": 8,
            "candidate_order": ["U", "UA", "CB"],
        },
        "code": {
            "cb_class_weight_artifact_sha256": sha256_file(class_weights),
        },
        "output": records_meta,
    }
    manifest_path = tmp_path / "synthetic-manifest.json"
    _write_json(manifest_path, manifest)
    contract = {
        "version": CONTRACT_VERSION,
        "status": "locked_before_candidate_aggregation",
        "role": SYNTHETIC_ROLE,
        "selection_boundary": {
            "candidate_aggregate_metrics_calculated": False,
            "candidate_rankings_calculated": False,
            "acceptance_gates_applied": False,
            "winner_selected": False,
        },
        "inputs": {
            "candidate_replay_manifest": {
                "path": manifest_path.name,
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            },
            "candidate_replay_records_identity_from_manifest_only": {
                "path": records.name,
                "bytes": records_meta["bytes"],
                "sha256": records_meta["sha256"],
            },
        },
        "lineage": {
            "active_groups": records_meta["jsonl_group_records"],
            "active_completions": records_meta["completion_records"],
            "ordered_sku_sha256": records_meta["ordered_sku_sha256"],
            "ordered_rollout_key_sha256": records_meta[
                "ordered_rollout_key_sha256"
            ],
        },
    }
    contract_path = tmp_path / "synthetic-contract.json"
    _write_json(contract_path, contract)
    return {
        "records": records,
        "manifest": manifest_path,
        "contract": contract_path,
        "class_weights": class_weights,
    }


def _add_full_scope_fixture(
    tmp_path: Path,
    files: dict[str, Path],
    *,
    groups_count: int = 3,
) -> dict[str, Path]:
    groups = []
    for position in range(groups_count):
        group = deepcopy(_group())
        sku_id = f"synthetic-full-sku-{position}"
        group["group_position"] = position
        group["sku_id"] = sku_id
        for completion in group["completions"]:
            completion["source_rollout"]["sku_id"] = sku_id
        groups.append(group)
    records = tmp_path / "synthetic-full-records.jsonl.gz"
    output = write_replay_groups(records, groups, repo_root=tmp_path)
    entries = [
        {
            "field_name": "colour_primary",
            "class_name": "synthetic_class",
            "active_pool_support": 0,
            "full_training_observations": 2,
            "affected_products": 2,
            "weight": 2.0,
        }
    ]
    import hashlib

    ledger_sha256 = hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    active_manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    active_groups = active_manifest["output"]["jsonl_group_records"]
    extension = {
        "version": CB_DIAGNOSTIC_EXTENSION_VERSION,
        "role": "gold-only synthetic diagnostic extension",
        "base_artifact_sha256": sha256_file(files["class_weights"]),
        "policy": {
            "weight": 2.0,
            "active_weights_changed": False,
            "full_training_support_used_to_retune_weights": False,
        },
        "missing_attribute_class_pairs": 1,
        "missing_class_observations": 2,
        "affected_training_products": 2,
        "affected_active_products": 0,
        "ordered_entry_ledger_sha256": ledger_sha256,
        "entries": entries,
        "candidate_completion_rewards_calculated": False,
        "candidate_aggregates_calculated": False,
    }
    manifest = {
        "version": FULL_REPLAY_VERSION,
        "status": "raw_evidence_published",
        "role": FULL_SYNTHETIC_ROLE,
        "selection_boundary": {
            "aggregate_candidate_comparison_calculated": False,
            "candidate_rankings_calculated": False,
            "acceptance_thresholds_applied": False,
            "winner_selected": False,
            "model_generation_performed": False,
            "validation_data_used": False,
            "probe_100_used": False,
            "legacy_frozen_300_used": False,
        },
        "record_contract": {
            "group_size": 8,
            "candidate_order": ["U", "UA", "CB"],
            "cb_diagnostic_extension_included": True,
        },
        "code": {
            "cb_class_weight_artifact_sha256": sha256_file(files["class_weights"]),
        },
        "output": output,
        "cb_diagnostic_extension": extension,
        "integrity": {
            "training_groups": groups_count,
            "completion_records": groups_count * 8,
            "active_groups_included": active_groups,
            "additional_training_groups": groups_count - active_groups,
            "ordered_sku_sha256": output["ordered_sku_sha256"],
            "ordered_rollout_key_sha256": output["ordered_rollout_key_sha256"],
            "role": FULL_SYNTHETIC_ROLE,
            "training_validation_sku_overlap": 0,
            "training_validation_family_overlap": 0,
            "candidate_aggregates_calculated": False,
            "model_generation_performed": False,
            "cb_active_weights_changed": False,
            "cb_extension_ledger_sha256": ledger_sha256,
        },
    }
    manifest_path = tmp_path / "synthetic-full-manifest.json"
    _write_json(manifest_path, manifest)
    files = dict(files)
    files["full_records"] = records
    files["full_manifest"] = manifest_path
    return files


def _run(tmp_path: Path, files: dict[str, Path], output_name: str):
    return run_analysis(
        repo_root=tmp_path,
        manifest_path=files["manifest"],
        records_path=files["records"],
        contract_path=files["contract"],
        class_weights_path=files["class_weights"],
        output_path=tmp_path / output_name,
        test_mode=True,
        bootstrap_seed=17,
        bootstrap_replicates=25,
        confidence=0.95,
    )


def test_synthetic_streaming_run_verifies_lineage_and_publishes_atomically(tmp_path):
    files = _fixture(tmp_path)
    output = tmp_path / "analysis.json"
    artifact = _run(tmp_path, files, output.name)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["version"] == VERSION
    assert artifact["status"] == "synthetic_orchestration_completed"
    assert artifact["lineage"]["groups"] == 2
    assert artifact["lineage"]["completions"] == 16
    assert artifact["lineage"]["groups_streamed_once"] is True
    assert artifact["lineage"]["groups_adapted_once"] is True
    assert artifact["inputs"]["all_identities_verified_before_gzip_open"] is True
    assert artifact["selection_boundary"] == {
        "candidate_aggregate_metrics_calculated": True,
        "real_candidate_replay_used": False,
        "acceptance_gates_applied": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
    }
    assert artifact["analysis_core"]["groups"] == 2
    assert artifact["contribution_and_segment_diagnostics"]["dominance"][
        "groups"
    ] == 2


def test_synthetic_orchestration_is_byte_deterministic(tmp_path):
    files = _fixture(tmp_path)
    _run(tmp_path, files, "first.json")
    _run(tmp_path, files, "second.json")
    assert (tmp_path / "first.json").read_bytes() == (
        tmp_path / "second.json"
    ).read_bytes()


def test_build_returns_complete_artifact_without_publication(tmp_path):
    files = _fixture(tmp_path)
    before = set(tmp_path.iterdir())
    artifact = build_analysis_artifact(
        repo_root=tmp_path,
        manifest_path=files["manifest"],
        records_path=files["records"],
        contract_path=files["contract"],
        class_weights_path=files["class_weights"],
        test_mode=True,
        bootstrap_seed=17,
        bootstrap_replicates=25,
        confidence=0.95,
    )

    assert artifact["status"] == "synthetic_orchestration_completed"
    assert artifact["lineage"]["groups"] == 2
    assert set(tmp_path.iterdir()) == before


def test_active_preflight_stops_before_gzip_open_or_adaptation(tmp_path, monkeypatch):
    files = _fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("active preflight must not decompress or adapt replay")

    monkeypatch.setattr("training.run2_analysis_orchestrator.gzip.open", forbidden)
    monkeypatch.setattr(
        "training.run2_analysis_orchestrator.adapt_replay_group", forbidden
    )
    report = run_active_preflight(
        repo_root=tmp_path,
        manifest_path=files["manifest"],
        records_path=files["records"],
        contract_path=files["contract"],
        class_weights_path=files["class_weights"],
        test_mode=True,
        bootstrap_seed=17,
        bootstrap_replicates=25,
    )

    assert report["status"] == "synthetic_active_preflight_passed"
    assert report["lineage"]["groups"] == 2
    assert report["selection_boundary"] == {
        "replay_gzip_decompressed": False,
        "replay_records_parsed": False,
        "candidate_aggregate_metrics_calculated": False,
        "acceptance_gates_applied": False,
        "candidate_rankings_calculated": False,
        "winner_selected": False,
        "artifact_published": False,
    }


def test_preflight_stops_before_gzip_open_adapter_or_publication(tmp_path, monkeypatch):
    files = _fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not decompress or adapt replay records")

    monkeypatch.setattr("training.run2_analysis_orchestrator.gzip.open", forbidden)
    monkeypatch.setattr(
        "training.run2_analysis_orchestrator.adapt_replay_group", forbidden
    )
    report = run_preflight(
        repo_root=tmp_path,
        manifest_path=files["manifest"],
        records_path=files["records"],
        contract_path=files["contract"],
        class_weights_path=files["class_weights"],
        test_mode=True,
        bootstrap_seed=17,
        bootstrap_replicates=25,
    )

    assert report["status"] == "synthetic_preflight_passed"
    assert report["inputs"]["all_identities_verified_before_gzip_open"] is True
    assert report["selection_boundary"] == {
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
    }
    assert set(tmp_path.iterdir()) == set(files.values())


def test_preflight_verifies_distinct_full_scope_and_cb_ledger_without_gzip(
    tmp_path, monkeypatch
):
    files = _add_full_scope_fixture(tmp_path, _fixture(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not decompress or adapt either replay")

    monkeypatch.setattr("training.run2_analysis_orchestrator.gzip.open", forbidden)
    monkeypatch.setattr(
        "training.run2_analysis_orchestrator.adapt_replay_group", forbidden
    )
    report = run_preflight(
        repo_root=tmp_path,
        manifest_path=files["manifest"],
        records_path=files["records"],
        contract_path=files["contract"],
        class_weights_path=files["class_weights"],
        full_manifest_path=files["full_manifest"],
        full_records_path=files["full_records"],
        test_mode=True,
        bootstrap_seed=17,
        bootstrap_replicates=25,
    )

    full = report["full_training_replay"]
    assert full["role"] == FULL_SYNTHETIC_ROLE
    assert full["groups"] == 3
    assert full["active_groups_included"] == 2
    assert full["additional_training_groups"] == 1
    assert full["scope_is_distinct_from_active_replay"] is True
    assert full["all_identities_verified_before_gzip_open"] is True
    assert full["cb_extension"]["ledger_hash_recomputed"] is True
    assert report["selection_boundary"]["gate_g10_calculated"] is False


def test_full_scope_paths_roles_hashes_and_ledger_fail_closed(tmp_path):
    files = _add_full_scope_fixture(tmp_path, _fixture(tmp_path))
    kwargs = {
        "repo_root": tmp_path,
        "manifest_path": files["manifest"],
        "records_path": files["records"],
        "contract_path": files["contract"],
        "class_weights_path": files["class_weights"],
        "full_manifest_path": files["full_manifest"],
        "full_records_path": files["full_records"],
        "test_mode": True,
        "bootstrap_seed": 17,
        "bootstrap_replicates": 25,
    }

    with pytest.raises(ValueError, match="records paths must be distinct"):
        run_preflight(**(kwargs | {"full_records_path": files["records"]}))

    original = json.loads(files["full_manifest"].read_text(encoding="utf-8"))
    wrong_role = deepcopy(original)
    wrong_role["role"] = SYNTHETIC_ROLE
    _write_json(files["full_manifest"], wrong_role)
    with pytest.raises(ValueError, match="full-training manifest role"):
        run_preflight(**kwargs)

    wrong_ledger = deepcopy(original)
    wrong_ledger["cb_diagnostic_extension"]["entries"][0][
        "class_name"
    ] = "tampered_class"
    _write_json(files["full_manifest"], wrong_ledger)
    with pytest.raises(ValueError, match="ledger SHA-256 mismatch"):
        run_preflight(**kwargs)

    _write_json(files["full_manifest"], original)
    files["full_records"].write_bytes(files["full_records"].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="byte count mismatch"):
        run_preflight(**kwargs)


def test_production_preflight_requires_full_scope_pair(tmp_path):
    files = _fixture(tmp_path)
    with pytest.raises(ValueError, match="requires the full-training replay pair"):
        run_preflight(
            repo_root=tmp_path,
            manifest_path=files["manifest"],
            records_path=files["records"],
            contract_path=files["contract"],
            class_weights_path=files["class_weights"],
            test_mode=False,
        )


def test_published_full_scope_identity_constants_match_real_artifacts():
    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / "runs/grpo-run2-full-training-candidate-replay-manifest.json"
    records = repo_root / "runs/grpo-run2-full-training-candidate-replay-records.jsonl.gz"

    assert manifest.stat().st_size == EXPECTED_FULL_MANIFEST_BYTES
    assert sha256_file(manifest) == EXPECTED_FULL_MANIFEST_SHA256
    assert records.stat().st_size == EXPECTED_FULL_RECORDS_BYTES
    assert sha256_file(records) == EXPECTED_FULL_RECORDS_SHA256


def test_hash_failure_happens_before_decompression_or_adaptation(tmp_path, monkeypatch):
    files = _fixture(tmp_path)
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = "0" * 64
    _write_json(files["manifest"], manifest)
    contract = json.loads(files["contract"].read_text(encoding="utf-8"))
    contract["inputs"]["candidate_replay_manifest"] = {
        "path": files["manifest"].name,
        "bytes": files["manifest"].stat().st_size,
        "sha256": sha256_file(files["manifest"]),
    }
    contract["inputs"]["candidate_replay_records_identity_from_manifest_only"][
        "sha256"
    ] = "0" * 64
    _write_json(files["contract"], contract)

    def forbidden(*args, **kwargs):
        raise AssertionError("adapter must not run after preflight hash failure")

    monkeypatch.setattr(
        "training.run2_analysis_orchestrator.adapt_replay_group", forbidden
    )
    with pytest.raises(ValueError, match="replay records SHA-256 mismatch"):
        _run(tmp_path, files, "should-not-exist.json")
    assert not (tmp_path / "should-not-exist.json").exists()


def test_stream_rejects_order_drift_and_leaves_no_partial_output(tmp_path):
    files = _fixture(tmp_path)
    # Rebuild a validly hashed gzip whose internal group positions are reversed.
    bad_groups = []
    for position in range(2):
        group = deepcopy(_group())
        sku_id = f"bad-{position}"
        group["group_position"] = 1 - position
        group["sku_id"] = sku_id
        for completion in group["completions"]:
            completion["source_rollout"]["sku_id"] = sku_id
        bad_groups.append(group)
    bad_records = tmp_path / "bad-order.jsonl.gz"
    # Bypass the production writer's own order check to test the reader gate.
    import gzip

    with gzip.GzipFile(filename="", mode="wb", fileobj=bad_records.open("wb"), mtime=0) as handle:
        for group in bad_groups:
            handle.write((json.dumps(group, sort_keys=True) + "\n").encode("utf-8"))
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    manifest["output"]["bytes"] = bad_records.stat().st_size
    manifest["output"]["sha256"] = sha256_file(bad_records)
    files["records"] = bad_records
    _write_json(files["manifest"], manifest)
    contract = json.loads(files["contract"].read_text(encoding="utf-8"))
    contract["inputs"]["candidate_replay_manifest"] = {
        "path": files["manifest"].name,
        "bytes": files["manifest"].stat().st_size,
        "sha256": sha256_file(files["manifest"]),
    }
    contract["inputs"]["candidate_replay_records_identity_from_manifest_only"] = {
        "path": bad_records.name,
        "bytes": bad_records.stat().st_size,
        "sha256": sha256_file(bad_records),
    }
    _write_json(files["contract"], contract)

    with pytest.raises(ValueError, match="group position drifted"):
        _run(tmp_path, files, "partial.json")
    assert not (tmp_path / "partial.json").exists()


def test_modes_and_output_collisions_fail_closed(tmp_path):
    files = _fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest role must be"):
        run_analysis(
            repo_root=tmp_path,
            manifest_path=files["manifest"],
            records_path=files["records"],
            contract_path=files["contract"],
            class_weights_path=files["class_weights"],
            output_path=tmp_path / "production.json",
            test_mode=False,
        )

    existing = tmp_path / "existing.json"
    existing.write_text("user data\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output already exists"):
        _run(tmp_path, files, existing.name)
    assert existing.read_text(encoding="utf-8") == "user data\n"


def test_cli_has_no_implicit_analysis_input_or_output_paths():
    with pytest.raises(SystemExit):
        parse_args([])
    args = parse_args(
        [
            "--manifest", "manifest.json",
            "--records", "records.jsonl.gz",
            "--contract", "contract.json",
            "--class-weights", "weights.json",
            "--full-manifest", "full-manifest.json",
            "--full-records", "full-records.jsonl.gz",
            "--preflight-only",
        ]
    )
    assert args.preflight_only is True
    assert args.output is None
    assert args.full_manifest == "full-manifest.json"
    assert args.full_records == "full-records.jsonl.gz"
