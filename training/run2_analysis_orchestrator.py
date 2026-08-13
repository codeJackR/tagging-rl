#!/usr/bin/env python3
"""Manifest-verified streaming orchestration for GRPO Run 2 analysis.

All paths are explicit. Before decompressing a replay, this module validates
the manifest, comparison contract, class-weight artifact, byte counts, and
SHA-256 identities. It then streams each group once, adapts it, verifies
canonical lineage, composes the locked in-memory analyses, and publishes one
exclusive atomic JSON artifact. Test mode is restricted to manifests explicitly
labeled as synthetic fixtures.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.analyze_run2_candidates import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    analyze_group_observations,
)
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.run2_comparison_contract import (
    EXPECTED_ACTIVE_COMPLETIONS,
    EXPECTED_ACTIVE_GROUPS,
    EXPECTED_CANDIDATE_REPLAY_VERSION,
    EXPECTED_GROUP_SIZE,
    VERSION as COMPARISON_CONTRACT_VERSION,
)
from training.replay_run2_full_training_candidates import (
    CB_DIAGNOSTIC_EXTENSION_VERSION,
    EXPECTED_ACTIVE_GROUPS as EXPECTED_FULL_SCOPE_ACTIVE_GROUPS,
    EXPECTED_CB_ZERO_SUPPORT_ENTRIES,
    EXPECTED_CB_ZERO_SUPPORT_PRODUCTS,
    EXPECTED_COMPLETIONS as EXPECTED_FULL_COMPLETIONS,
    EXPECTED_ORDERED_KEY_SHA256 as EXPECTED_FULL_ORDERED_KEY_SHA256,
    EXPECTED_ORDERED_SKU_SHA256 as EXPECTED_FULL_ORDERED_SKU_SHA256,
    EXPECTED_TRAINING_GROUPS,
    FULL_TRAINING_ROLE,
    MANIFEST_VERSION as FULL_TRAINING_MANIFEST_VERSION,
)
from training.run2_replay_adapter import (
    AdaptedReplayGroup,
    adapt_replay_group,
    build_class_support_lookup,
)
from training.run2_segment_summaries import summarize_in_memory_diagnostics


VERSION = "grpo-run2-analysis-orchestrator-v1"
SYNTHETIC_ROLE = "synthetic_test_fixture"
FULL_SYNTHETIC_ROLE = "synthetic_full_training_test_fixture"
PRODUCTION_ROLE = "training_only_identical_group_candidate_replay_records"
EXPECTED_CLASS_WEIGHT_VERSION = "grpo-run2-cb-class-weights-v1"
EXPECTED_CB_EXTENSION_ROLE = (
    "gold-only full-training diagnostic extension to active CB lookup"
)
EXPECTED_FULL_MANIFEST_BYTES = 10_709
EXPECTED_FULL_MANIFEST_SHA256 = (
    "ad7f4b8b3749062b73b7b35f25c206cad8ef17ca55b248fbeff228e35a2bc9a0"
)
EXPECTED_FULL_RECORDS_BYTES = 4_168_170
EXPECTED_FULL_RECORDS_SHA256 = (
    "9b4a110910977b54e181c8f3d3452c555bdbb6c826e22bfec1c475045517fcf9"
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is invalid JSON: {exc}") from exc
    return _mapping(value, name)


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


def _ordered_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_identity(
    path: Path, metadata: Mapping[str, Any], name: str, *, repo_root: Path
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _file_metadata(path, repo_root)
    expected_bytes = metadata.get("bytes")
    expected_hash = metadata.get("sha256")
    if observed["bytes"] != expected_bytes:
        raise ValueError(
            f"{name} byte count mismatch: expected={expected_bytes}, "
            f"observed={observed['bytes']}"
        )
    if observed["sha256"] != expected_hash:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected={expected_hash}, "
            f"observed={observed['sha256']}"
        )
    return observed


def _false_boundary(manifest: Mapping[str, Any]) -> None:
    boundary = _mapping(manifest.get("selection_boundary"), "manifest selection_boundary")
    for key in (
        "aggregate_candidate_comparison_calculated",
        "candidate_rankings_calculated",
        "acceptance_thresholds_applied",
        "winner_selected",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"manifest boundary {key} must be explicitly false")


def _verify_cb_extension_ledger(
    *,
    manifest: Mapping[str, Any],
    class_weights_sha256: str,
    test_mode: bool,
) -> dict[str, Any]:
    """Validate the full-scope CB extension without calculating rewards."""
    extension = _mapping(
        manifest.get("cb_diagnostic_extension"),
        "full-training manifest CB diagnostic extension",
    )
    if extension.get("version") != CB_DIAGNOSTIC_EXTENSION_VERSION:
        raise ValueError("full-training CB extension version drifted")
    if not test_mode and extension.get("role") != EXPECTED_CB_EXTENSION_ROLE:
        raise ValueError("full-training CB extension role drifted")
    if extension.get("base_artifact_sha256") != class_weights_sha256:
        raise ValueError("full-training CB extension base artifact hash drifted")
    if extension.get("affected_active_products") != 0:
        raise ValueError("full-training CB extension affects active products")
    if extension.get("candidate_completion_rewards_calculated") is not False:
        raise ValueError("full-training CB extension used completion rewards")
    if extension.get("candidate_aggregates_calculated") is not False:
        raise ValueError("full-training CB extension used candidate aggregates")

    policy = _mapping(extension.get("policy"), "full-training CB extension policy")
    if policy.get("active_weights_changed") is not False:
        raise ValueError("full-training CB extension changed active weights")
    if policy.get("full_training_support_used_to_retune_weights") is not False:
        raise ValueError("full-training CB extension retuned active weights")
    weight = policy.get("weight")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise ValueError("full-training CB extension policy weight is invalid")
    if not math.isfinite(float(weight)) or float(weight) <= 0:
        raise ValueError("full-training CB extension policy weight is invalid")

    entries = extension.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("full-training CB extension ledger is missing")
    previous_key: tuple[str, str] | None = None
    observed_contract_entries: list[tuple[str, str, int]] = []
    observations = 0
    for entry in entries:
        entry = _mapping(entry, "full-training CB extension ledger entry")
        field_name = entry.get("field_name")
        class_name = entry.get("class_name")
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("full-training CB extension field name is invalid")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("full-training CB extension class name is invalid")
        key = (field_name, class_name)
        if previous_key is not None and key <= previous_key:
            raise ValueError("full-training CB extension ledger is not uniquely ordered")
        previous_key = key
        if entry.get("active_pool_support") != 0:
            raise ValueError("full-training CB extension entry has active support")
        if entry.get("weight") != weight:
            raise ValueError("full-training CB extension entry weight drifted")
        entry_observations = entry.get("full_training_observations")
        affected_products = entry.get("affected_products")
        if not isinstance(entry_observations, int) or entry_observations <= 0:
            raise ValueError("full-training CB extension observation count is invalid")
        if (
            not isinstance(affected_products, int)
            or affected_products <= 0
            or affected_products > entry_observations
        ):
            raise ValueError("full-training CB extension affected-product count is invalid")
        observations += entry_observations
        observed_contract_entries.append(
            (field_name, class_name, entry_observations)
        )

    if extension.get("missing_attribute_class_pairs") != len(entries):
        raise ValueError("full-training CB extension class-pair count drifted")
    if extension.get("missing_class_observations") != observations:
        raise ValueError("full-training CB extension observation total drifted")
    affected_training_products = extension.get("affected_training_products")
    if not isinstance(affected_training_products, int) or affected_training_products <= 0:
        raise ValueError("full-training CB extension affected-product total is invalid")

    observed_ledger_sha256 = hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if extension.get("ordered_entry_ledger_sha256") != observed_ledger_sha256:
        raise ValueError("full-training CB extension ledger SHA-256 mismatch")

    if not test_mode:
        if tuple(observed_contract_entries) != EXPECTED_CB_ZERO_SUPPORT_ENTRIES:
            raise ValueError("production full-training CB extension entries drifted")
        if affected_training_products != EXPECTED_CB_ZERO_SUPPORT_PRODUCTS:
            raise ValueError("production full-training CB affected-product total drifted")

    return {
        "version": extension["version"],
        "entries": len(entries),
        "observations": observations,
        "affected_training_products": affected_training_products,
        "affected_active_products": 0,
        "ordered_entry_ledger_sha256": observed_ledger_sha256,
        "active_weights_changed": False,
        "candidate_aggregates_calculated": False,
        "ledger_hash_recomputed": True,
    }


def _verify_full_scope_preflight(
    *,
    repo_root: Path,
    active_manifest: Mapping[str, Any],
    active_manifest_path: Path,
    active_records_path: Path,
    full_manifest_path: Path,
    full_records_path: Path,
    class_weights_sha256: str,
    test_mode: bool,
) -> dict[str, Any]:
    """Verify the broader replay as a distinct raw-evidence scope."""
    if full_manifest_path == active_manifest_path:
        raise ValueError("active and full-training manifest paths must be distinct")
    if full_records_path == active_records_path:
        raise ValueError("active and full-training records paths must be distinct")
    if not full_manifest_path.is_file():
        raise FileNotFoundError(full_manifest_path)
    full_manifest = _load_json(full_manifest_path, "full-training replay manifest")
    if full_manifest.get("version") != FULL_TRAINING_MANIFEST_VERSION:
        raise ValueError("unexpected full-training replay manifest version")
    if full_manifest.get("status") != "raw_evidence_published":
        raise ValueError("full-training replay manifest is not raw published evidence")
    expected_role = FULL_SYNTHETIC_ROLE if test_mode else FULL_TRAINING_ROLE
    if full_manifest.get("role") != expected_role:
        raise ValueError(f"full-training manifest role must be {expected_role!r}")
    if full_manifest.get("role") == active_manifest.get("role"):
        raise ValueError("active and full-training replay roles must be distinct")
    _false_boundary(full_manifest)

    boundary = _mapping(
        full_manifest.get("selection_boundary"),
        "full-training manifest selection_boundary",
    )
    for key in (
        "model_generation_performed",
        "validation_data_used",
        "probe_100_used",
        "legacy_frozen_300_used",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"full-training manifest boundary {key} must be false")

    record_contract = _mapping(
        full_manifest.get("record_contract"),
        "full-training manifest record_contract",
    )
    if record_contract.get("group_size") != EXPECTED_GROUP_SIZE:
        raise ValueError("full-training manifest group size must be eight")
    if tuple(record_contract.get("candidate_order", ())) != ("U", "UA", "CB"):
        raise ValueError("full-training manifest candidate order drifted")
    if record_contract.get("cb_diagnostic_extension_included") is not True:
        raise ValueError("full-training manifest omits the CB diagnostic extension")

    code = _mapping(full_manifest.get("code"), "full-training manifest code")
    if code.get("cb_class_weight_artifact_sha256") != class_weights_sha256:
        raise ValueError("full-training replay class-weight artifact hash drifted")
    output = _mapping(full_manifest.get("output"), "full-training manifest output")
    full_records_observed = _verify_identity(
        full_records_path,
        output,
        "full-training replay records",
        repo_root=repo_root,
    )
    full_manifest_observed = _file_metadata(full_manifest_path, repo_root)
    if not test_mode:
        expected_published_identities = (
            EXPECTED_FULL_MANIFEST_BYTES,
            EXPECTED_FULL_MANIFEST_SHA256,
            EXPECTED_FULL_RECORDS_BYTES,
            EXPECTED_FULL_RECORDS_SHA256,
        )
        observed_published_identities = (
            full_manifest_observed["bytes"],
            full_manifest_observed["sha256"],
            full_records_observed["bytes"],
            full_records_observed["sha256"],
        )
        if observed_published_identities != expected_published_identities:
            raise ValueError("published full-training replay identities drifted")

    groups = output.get("jsonl_group_records")
    completions = output.get("completion_records")
    if not isinstance(groups, int) or groups <= 0:
        raise ValueError("full-training replay group count must be positive")
    if completions != groups * EXPECTED_GROUP_SIZE:
        raise ValueError("full-training replay completion count is not eight per group")
    active_output = _mapping(active_manifest.get("output"), "active manifest output")
    active_groups = active_output.get("jsonl_group_records")
    if not isinstance(active_groups, int) or groups <= active_groups:
        raise ValueError("full-training scope must strictly contain the active scope")
    if output.get("sha256") == active_output.get("sha256"):
        raise ValueError("active and full-training replay identities must be distinct")

    integrity = _mapping(full_manifest.get("integrity"), "full-training integrity")
    expected_integrity = {
        "training_groups": groups,
        "completion_records": completions,
        "active_groups_included": active_groups,
        "additional_training_groups": groups - active_groups,
        "ordered_sku_sha256": output.get("ordered_sku_sha256"),
        "ordered_rollout_key_sha256": output.get("ordered_rollout_key_sha256"),
        "role": expected_role,
    }
    for key, expected in expected_integrity.items():
        if integrity.get(key) != expected:
            raise ValueError(f"full-training integrity {key} drifted")
    for key in (
        "training_validation_sku_overlap",
        "training_validation_family_overlap",
    ):
        if integrity.get(key) != 0:
            raise ValueError(f"full-training integrity {key} must be zero")
    for key in ("candidate_aggregates_calculated", "model_generation_performed"):
        if integrity.get(key) is not False:
            raise ValueError(f"full-training integrity {key} must be false")
    if integrity.get("cb_active_weights_changed") is not False:
        raise ValueError("full-training integrity claims active CB weights changed")

    cb_ledger = _verify_cb_extension_ledger(
        manifest=full_manifest,
        class_weights_sha256=class_weights_sha256,
        test_mode=test_mode,
    )
    if integrity.get("cb_extension_ledger_sha256") != cb_ledger[
        "ordered_entry_ledger_sha256"
    ]:
        raise ValueError("full-training integrity and CB ledger hashes disagree")

    if not test_mode:
        production_expected = (
            EXPECTED_TRAINING_GROUPS,
            EXPECTED_FULL_COMPLETIONS,
            EXPECTED_FULL_SCOPE_ACTIVE_GROUPS,
            EXPECTED_FULL_ORDERED_SKU_SHA256,
            EXPECTED_FULL_ORDERED_KEY_SHA256,
        )
        observed = (
            groups,
            completions,
            active_groups,
            output.get("ordered_sku_sha256"),
            output.get("ordered_rollout_key_sha256"),
        )
        if observed != production_expected:
            raise ValueError("production full-training replay scope drifted")

    return {
        "manifest": full_manifest_observed,
        "records": full_records_observed,
        "role": expected_role,
        "groups": groups,
        "completions": completions,
        "active_groups_included": active_groups,
        "additional_training_groups": groups - active_groups,
        "ordered_sku_sha256": output["ordered_sku_sha256"],
        "ordered_rollout_key_sha256": output["ordered_rollout_key_sha256"],
        "cb_extension": cb_ledger,
        "scope_is_distinct_from_active_replay": True,
        "all_identities_verified_before_gzip_open": True,
    }


def _verify_preflight(
    *,
    repo_root: Path,
    manifest_path: Path,
    records_path: Path,
    contract_path: Path,
    class_weights_path: Path,
    test_mode: bool,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    confidence: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    """Verify every small control file and compressed identity before gzip open."""
    for path in (manifest_path, contract_path, class_weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _load_json(manifest_path, "candidate replay manifest")
    contract = _load_json(contract_path, "comparison contract")
    class_weights = _load_json(class_weights_path, "class-weight artifact")

    if manifest.get("version") != EXPECTED_CANDIDATE_REPLAY_VERSION:
        raise ValueError("unexpected candidate replay manifest version")
    if manifest.get("status") != "raw_evidence_published":
        raise ValueError("candidate replay manifest is not raw published evidence")
    expected_role = SYNTHETIC_ROLE if test_mode else PRODUCTION_ROLE
    if manifest.get("role") != expected_role:
        raise ValueError(
            f"manifest role must be {expected_role!r} for "
            f"{'test' if test_mode else 'production'} mode"
        )
    _false_boundary(manifest)
    output = _mapping(manifest.get("output"), "manifest output")
    record_contract = _mapping(manifest.get("record_contract"), "manifest record_contract")
    if record_contract.get("group_size") != 8:
        raise ValueError("manifest group size must be eight")
    if tuple(record_contract.get("candidate_order", ())) != ("U", "UA", "CB"):
        raise ValueError("manifest candidate order drifted")

    if contract.get("version") != COMPARISON_CONTRACT_VERSION:
        raise ValueError("unexpected comparison contract version")
    if contract.get("status") != "locked_before_candidate_aggregation":
        raise ValueError("comparison contract is not locked before aggregation")
    contract_boundary = _mapping(
        contract.get("selection_boundary"), "contract selection_boundary"
    )
    for key in (
        "candidate_aggregate_metrics_calculated",
        "candidate_rankings_calculated",
        "acceptance_gates_applied",
        "winner_selected",
    ):
        if contract_boundary.get(key) is not False:
            raise ValueError(f"comparison contract boundary {key} must be false")

    contract_inputs = _mapping(contract.get("inputs"), "contract inputs")
    manifest_contract_metadata = _mapping(
        contract_inputs.get("candidate_replay_manifest"),
        "contract candidate_replay_manifest",
    )
    manifest_observed = _verify_identity(
        manifest_path,
        manifest_contract_metadata,
        "candidate replay manifest",
        repo_root=repo_root,
    )
    record_contract_metadata = _mapping(
        contract_inputs.get("candidate_replay_records_identity_from_manifest_only"),
        "contract candidate replay records identity",
    )
    if (
        record_contract_metadata.get("bytes") != output.get("bytes")
        or record_contract_metadata.get("sha256") != output.get("sha256")
    ):
        raise ValueError("contract and manifest disagree on replay-record identity")
    records_observed = _verify_identity(
        records_path, output, "candidate replay records", repo_root=repo_root
    )

    if class_weights.get("version") != EXPECTED_CLASS_WEIGHT_VERSION:
        raise ValueError("unexpected class-weight artifact version")
    class_observed = _file_metadata(class_weights_path, repo_root)
    expected_class_hash = _mapping(manifest.get("code"), "manifest code").get(
        "cb_class_weight_artifact_sha256"
    )
    if class_observed["sha256"] != expected_class_hash:
        raise ValueError("class-weight artifact SHA-256 differs from replay manifest")

    expected_groups = output.get("jsonl_group_records")
    expected_completions = output.get("completion_records")
    if not isinstance(expected_groups, int) or expected_groups <= 0:
        raise ValueError("manifest output group count must be positive")
    if expected_completions != expected_groups * 8:
        raise ValueError("manifest completion count is not eight per group")
    lineage = _mapping(contract.get("lineage"), "contract lineage")
    for name, expected, observed in (
        ("groups", expected_groups, lineage.get("active_groups")),
        ("completions", expected_completions, lineage.get("active_completions")),
        (
            "ordered SKU hash",
            output.get("ordered_sku_sha256"),
            lineage.get("ordered_sku_sha256"),
        ),
        (
            "ordered rollout-key hash",
            output.get("ordered_rollout_key_sha256"),
            lineage.get("ordered_rollout_key_sha256"),
        ),
    ):
        if expected != observed:
            raise ValueError(f"contract/manifest {name} mismatch")

    if test_mode:
        if contract.get("role") != SYNTHETIC_ROLE:
            raise ValueError("test-mode comparison contract must be a synthetic fixture")
    else:
        if (expected_groups, expected_completions) != (
            EXPECTED_ACTIVE_GROUPS,
            EXPECTED_ACTIVE_COMPLETIONS,
        ):
            raise ValueError("production replay counts differ from the locked active scope")
        uncertainty = _mapping(
            contract.get("uncertainty_contract"), "contract uncertainty_contract"
        )
        locked_settings = (
            uncertainty.get("seed"),
            uncertainty.get("replicates"),
            uncertainty.get("confidence_level"),
        )
        requested_settings = (
            bootstrap_seed,
            bootstrap_replicates,
            confidence,
        )
        if requested_settings != locked_settings or requested_settings != (
            BOOTSTRAP_SEED,
            BOOTSTRAP_REPLICATES,
            CONFIDENCE_LEVEL,
        ):
            raise ValueError("production bootstrap settings differ from the locked contract")

    return manifest, contract, class_weights, {
        "manifest": manifest_observed,
        "records": records_observed,
        "comparison_contract": _file_metadata(contract_path, repo_root),
        "class_weights": class_observed,
        "all_identities_verified_before_gzip_open": True,
    }


def _stream_adapted_groups(
    *,
    records_path: Path,
    manifest: Mapping[str, Any],
    class_weights: Mapping[str, Any],
) -> tuple[list[AdaptedReplayGroup], dict[str, Any]]:
    output = _mapping(manifest["output"], "manifest output")
    expected_groups = output["jsonl_group_records"]
    expected_completions = output["completion_records"]
    class_supports = build_class_support_lookup(class_weights)
    adapted: list[AdaptedReplayGroup] = []
    sku_ids: list[str] = []
    rollout_keys: list[str] = []
    seen_skus: set[str] = set()

    try:
        with gzip.open(records_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank replay record at line {line_number}")
                try:
                    group = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid replay JSON at line {line_number}: {exc}"
                    ) from exc
                group = _mapping(group, f"replay line {line_number}")
                expected_position = line_number - 1
                if group.get("group_position") != expected_position:
                    raise ValueError(
                        f"replay group position drifted at line {line_number}"
                    )
                sku_id = group.get("sku_id")
                if not isinstance(sku_id, str) or not sku_id:
                    raise ValueError(f"invalid replay SKU at line {line_number}")
                if sku_id in seen_skus:
                    raise ValueError(f"duplicate replay SKU: {sku_id}")
                seen_skus.add(sku_id)
                item = adapt_replay_group(group, class_supports=class_supports)
                adapted.append(item)
                sku_ids.append(sku_id)
                rollout_keys.extend(
                    f"{sku_id}\t{index}" for index in range(EXPECTED_GROUP_SIZE)
                )
    except (OSError, EOFError) as exc:
        raise ValueError(f"candidate replay gzip could not be decoded: {exc}") from exc

    observed_groups = len(adapted)
    observed_completions = observed_groups * EXPECTED_GROUP_SIZE
    observed_sku_hash = _ordered_hash(sku_ids)
    observed_key_hash = _ordered_hash(rollout_keys)
    if observed_groups != expected_groups:
        raise ValueError(
            f"replay group count mismatch: expected={expected_groups}, "
            f"observed={observed_groups}"
        )
    if observed_completions != expected_completions:
        raise ValueError("replay completion count mismatch")
    if observed_sku_hash != output.get("ordered_sku_sha256"):
        raise ValueError("ordered replay SKU hash mismatch")
    if observed_key_hash != output.get("ordered_rollout_key_sha256"):
        raise ValueError("ordered replay rollout-key hash mismatch")
    return adapted, {
        "groups": observed_groups,
        "completions": observed_completions,
        "unique_skus": len(seen_skus),
        "ordered_sku_sha256": observed_sku_hash,
        "ordered_rollout_key_sha256": observed_key_hash,
        "groups_streamed_once": True,
        "groups_adapted_once": True,
    }


def run_preflight(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    records_path: str | Path,
    contract_path: str | Path,
    class_weights_path: str | Path,
    full_manifest_path: str | Path | None = None,
    full_records_path: str | Path | None = None,
    test_mode: bool = False,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Verify both raw replay scopes and stop before gzip decompression."""
    repo_root = Path(repo_root).resolve()
    if (full_manifest_path is None) != (full_records_path is None):
        raise ValueError(
            "full-training manifest and records paths must be provided together"
        )
    if not test_mode and full_manifest_path is None:
        raise ValueError("production preflight requires the full-training replay pair")
    paths = {
        "manifest": Path(manifest_path),
        "records": Path(records_path),
        "contract": Path(contract_path),
        "class_weights": Path(class_weights_path),
    }
    if full_manifest_path is not None and full_records_path is not None:
        paths["full_manifest"] = Path(full_manifest_path)
        paths["full_records"] = Path(full_records_path)
    paths = {
        name: path.resolve() if path.is_absolute() else (repo_root / path).resolve()
        for name, path in paths.items()
    }
    manifest, contract, _class_weights, identities = _verify_preflight(
        repo_root=repo_root,
        manifest_path=paths["manifest"],
        records_path=paths["records"],
        contract_path=paths["contract"],
        class_weights_path=paths["class_weights"],
        test_mode=test_mode,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
    )
    output = _mapping(manifest["output"], "manifest output")
    full_scope = None
    if "full_manifest" in paths:
        full_scope = _verify_full_scope_preflight(
            repo_root=repo_root,
            active_manifest=manifest,
            active_manifest_path=paths["manifest"],
            active_records_path=paths["records"],
            full_manifest_path=paths["full_manifest"],
            full_records_path=paths["full_records"],
            class_weights_sha256=identities["class_weights"]["sha256"],
            test_mode=test_mode,
        )
    return {
        "version": VERSION,
        "status": "synthetic_preflight_passed" if test_mode else "production_preflight_passed",
        "mode": (
            "synthetic_dual_replay"
            if test_mode and full_scope is not None
            else "synthetic_test"
            if test_mode
            else "locked_dual_replay"
        ),
        "role": manifest["role"],
        "inputs": identities,
        "lineage": {
            "groups": output["jsonl_group_records"],
            "completions": output["completion_records"],
            "ordered_sku_sha256": output["ordered_sku_sha256"],
            "ordered_rollout_key_sha256": output["ordered_rollout_key_sha256"],
            "contract_lineage_matches_manifest": True,
        },
        "full_training_replay": full_scope,
        "settings": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "confidence": confidence,
            "settings_locked_to_production_contract": not test_mode,
        },
        "comparison_contract_version": contract["version"],
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


def run_active_preflight(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    records_path: str | Path,
    contract_path: str | Path,
    class_weights_path: str | Path,
    test_mode: bool = False,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Verify only the active replay scope and stop before gzip decompression."""
    repo_root = Path(repo_root).resolve()
    paths = {
        "manifest": Path(manifest_path),
        "records": Path(records_path),
        "contract": Path(contract_path),
        "class_weights": Path(class_weights_path),
    }
    paths = {
        name: path.resolve() if path.is_absolute() else (repo_root / path).resolve()
        for name, path in paths.items()
    }
    manifest, contract, _class_weights, identities = _verify_preflight(
        repo_root=repo_root,
        manifest_path=paths["manifest"],
        records_path=paths["records"],
        contract_path=paths["contract"],
        class_weights_path=paths["class_weights"],
        test_mode=test_mode,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
    )
    output = _mapping(manifest["output"], "manifest output")
    return {
        "version": VERSION,
        "status": (
            "synthetic_active_preflight_passed"
            if test_mode
            else "production_active_preflight_passed"
        ),
        "mode": "synthetic_test" if test_mode else "locked_active_replay",
        "role": manifest["role"],
        "inputs": identities,
        "lineage": {
            "groups": output["jsonl_group_records"],
            "completions": output["completion_records"],
            "ordered_sku_sha256": output["ordered_sku_sha256"],
            "ordered_rollout_key_sha256": output["ordered_rollout_key_sha256"],
            "contract_lineage_matches_manifest": True,
        },
        "settings": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "confidence": confidence,
            "settings_locked_to_production_contract": not test_mode,
        },
        "comparison_contract_version": contract["version"],
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


def build_analysis_artifact(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    records_path: str | Path,
    contract_path: str | Path,
    class_weights_path: str | Path,
    test_mode: bool = False,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Verify, stream and build one complete artifact without publishing it."""
    repo_root = Path(repo_root).resolve()
    paths = {
        "manifest": Path(manifest_path),
        "records": Path(records_path),
        "contract": Path(contract_path),
        "class_weights": Path(class_weights_path),
    }
    paths = {
        name: path.resolve() if path.is_absolute() else (repo_root / path).resolve()
        for name, path in paths.items()
    }
    manifest, contract, class_weights, identities = _verify_preflight(
        repo_root=repo_root,
        manifest_path=paths["manifest"],
        records_path=paths["records"],
        contract_path=paths["contract"],
        class_weights_path=paths["class_weights"],
        test_mode=test_mode,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
    )
    adapted, lineage = _stream_adapted_groups(
        records_path=paths["records"],
        manifest=manifest,
        class_weights=class_weights,
    )
    core = analyze_group_observations(
        [group.observation for group in adapted],
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
    )
    diagnostics = summarize_in_memory_diagnostics(adapted)
    return {
        "version": VERSION,
        "status": (
            "synthetic_orchestration_completed"
            if test_mode
            else "aggregate_candidate_analysis_completed_pending_gates"
        ),
        "role": manifest["role"],
        "mode": "synthetic_test" if test_mode else "locked_active_replay",
        "selection_boundary": {
            "candidate_aggregate_metrics_calculated": True,
            "real_candidate_replay_used": not test_mode,
            "acceptance_gates_applied": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
        },
        "inputs": identities,
        "lineage": lineage,
        "settings": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "confidence": confidence,
            "settings_locked_to_production_contract": not test_mode,
        },
        "implementation": _file_metadata(Path(__file__).resolve(), repo_root),
        "comparison_contract_version": contract["version"],
        "analysis_core": core,
        "contribution_and_segment_diagnostics": diagnostics,
    }


def run_analysis(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    records_path: str | Path,
    contract_path: str | Path,
    class_weights_path: str | Path,
    output_path: str | Path,
    test_mode: bool = False,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Build and exclusively publish one complete analysis artifact."""
    repo_root = Path(repo_root).resolve()
    output = Path(output_path)
    output = output.resolve() if output.is_absolute() else (repo_root / output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    artifact = build_analysis_artifact(
        repo_root=repo_root,
        manifest_path=manifest_path,
        records_path=records_path,
        contract_path=contract_path,
        class_weights_path=class_weights_path,
        test_mode=test_mode,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
    )
    write_exclusive_atomic_json(output, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--class-weights", required=True)
    parser.add_argument("--full-manifest")
    parser.add_argument("--full-records")
    parser.add_argument("--output")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_LEVEL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        report = run_preflight(
            repo_root=args.repo_root,
            manifest_path=args.manifest,
            records_path=args.records,
            contract_path=args.contract,
            class_weights_path=args.class_weights,
            full_manifest_path=args.full_manifest,
            full_records_path=args.full_records,
            test_mode=args.test_mode,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            confidence=args.confidence,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not args.output:
        raise SystemExit("--output is required unless --preflight-only is used")
    artifact = run_analysis(
        repo_root=args.repo_root,
        manifest_path=args.manifest,
        records_path=args.records,
        contract_path=args.contract,
        class_weights_path=args.class_weights,
        output_path=args.output,
        test_mode=args.test_mode,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence=args.confidence,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "groups": artifact["lineage"]["groups"],
                "winner_selected": artifact["selection_boundary"]["winner_selected"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
