#!/usr/bin/env python3
"""Validate and iterate the full-training candidate replay scope.

This module deliberately stops before file publication. It selects the 3,240
authoritative SFT-training products from the already-locked k=8 rollout source,
proves the scope boundary, and delegates each group to the existing candidate
replay scorer. It never generates completions or aggregates candidate outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.build_cb_class_weights import label_class_keys
from training.replay_original_reward import ReplayInputs
from training.replay_run2_candidates import (
    CANDIDATES,
    DEFAULT_MANIFEST_OUTPUT as ACTIVE_MANIFEST_OUTPUT,
    DEFAULT_RECORDS_OUTPUT as ACTIVE_RECORDS_OUTPUT,
    build_replay_group,
    write_replay_groups,
)
from training.run2_rewards import (
    CB_CLASS_WEIGHT_ARTIFACT_SHA256,
    VERSION as REWARD_IMPLEMENTATION_VERSION,
    CBClassWeightLookup,
)
from training.reward_scale_contract import CLASS_WEIGHT_MAX
from training.score_difficulty import ROLLOUTS_PER_PROMPT
from training.split_sft import group_key
from verifier import Pack


VERSION = "grpo-run2-full-training-replay-builder-v1"
FULL_TRAINING_ROLE = (
    "full_authoritative_training_identical_group_candidate_replay_records"
)
EXPECTED_TRAINING_GROUPS = 3_240
EXPECTED_VALIDATION_GROUPS = 360
EXPECTED_ACTIVE_GROUPS = 1_438
EXPECTED_COMPLETIONS = EXPECTED_TRAINING_GROUPS * ROLLOUTS_PER_PROMPT
EXPECTED_ORDERED_SKU_SHA256 = (
    "05e22c09120a63f9936473fd1adf8bf7639545cbe2a22bdbb28b8ab2d74906ee"
)
EXPECTED_ORDERED_KEY_SHA256 = (
    "a6acc3446db2102b95fdec7fc798f731969a473b053bc23fe0e5a7a1d9851d59"
)
DEFAULT_RECORDS_OUTPUT = (
    "runs/grpo-run2-full-training-candidate-replay-records.jsonl.gz"
)
DEFAULT_MANIFEST_OUTPUT = (
    "runs/grpo-run2-full-training-candidate-replay-manifest.json"
)
MANIFEST_VERSION = "grpo-run2-full-training-candidate-replay-records-v1"
CB_DIAGNOSTIC_EXTENSION_VERSION = (
    "grpo-run2-cb-full-training-zero-active-support-extension-v1"
)
CB_ZERO_ACTIVE_SUPPORT_WEIGHT = CLASS_WEIGHT_MAX
EXPECTED_CB_ZERO_SUPPORT_ENTRIES = (
    ("closure", "buckle", 3),
    ("colour_primary", "metallic", 1),
    ("colour_primary", "orange", 10),
    ("details", "__not_applicable__", 4),
    ("details", "gathered", 12),
    ("details", "tiered", 5),
    ("garment_category", "jumpsuit", 2),
    ("neckline", "collarless", 1),
    ("occasion", "__not_applicable__", 3),
    ("pattern", "fair_isle", 1),
    ("silhouette", "fit_and_flare", 6),
    ("silhouette", "shift", 4),
    ("waistline", "low", 1),
)
EXPECTED_CB_ZERO_SUPPORT_PRODUCTS = 50


@dataclass(frozen=True)
class FullTrainingScopeContract:
    """Expected denominator and order for one replay scope."""

    training_groups: int
    validation_groups: int
    active_groups: int
    ordered_sku_sha256: str
    ordered_rollout_key_sha256: str
    role: str = FULL_TRAINING_ROLE

    @property
    def completion_records(self) -> int:
        return self.training_groups * ROLLOUTS_PER_PROMPT


PRODUCTION_SCOPE_CONTRACT = FullTrainingScopeContract(
    training_groups=EXPECTED_TRAINING_GROUPS,
    validation_groups=EXPECTED_VALIDATION_GROUPS,
    active_groups=EXPECTED_ACTIVE_GROUPS,
    ordered_sku_sha256=EXPECTED_ORDERED_SKU_SHA256,
    ordered_rollout_key_sha256=EXPECTED_ORDERED_KEY_SHA256,
)


@dataclass(frozen=True)
class CBDiagnosticExtensionContract:
    """Locked zero-active-support policy for the broader diagnostic scope."""

    expected_entries: tuple[tuple[str, str, int], ...]
    expected_affected_products: int
    fallback_weight: float = CB_ZERO_ACTIVE_SUPPORT_WEIGHT


PRODUCTION_CB_EXTENSION_CONTRACT = CBDiagnosticExtensionContract(
    expected_entries=EXPECTED_CB_ZERO_SUPPORT_ENTRIES,
    expected_affected_products=EXPECTED_CB_ZERO_SUPPORT_PRODUCTS,
)


@dataclass(frozen=True)
class FullTrainingCBExtension:
    """Immutable derived lookup plus its gold-only provenance ledger."""

    base_lookup: CBClassWeightLookup
    lookup: CBClassWeightLookup
    audit: Mapping[str, Any]


def ordered_sha256(values: Iterable[str]) -> str:
    """Hash ordered newline-delimited values using the published convention."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_within_repo(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("full-training replay outputs must stay inside repo root") from exc
    return resolved


def _display_path(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root))


def _file_metadata(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _portable(value: Any, *, repo_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _portable(item, repo_root=repo_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable(item, repo_root=repo_root) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            return str(Path(value).resolve().relative_to(repo_root))
        except ValueError:
            return value
    return value


def _require_unique(name: str, values: list[str]) -> set[str]:
    observed = set(values)
    if len(observed) != len(values):
        raise ValueError(f"{name} contains duplicate SKUs")
    return observed


def build_full_training_cb_extension(
    *,
    inputs: ReplayInputs,
    pack: Pack,
    base_lookup: CBClassWeightLookup,
    contract: CBDiagnosticExtensionContract = PRODUCTION_CB_EXTENSION_CONTRACT,
) -> FullTrainingCBExtension:
    """Add only gold classes with zero active-pool support at the locked cap.

    This is a diagnostic-scope extension. It never changes a published active
    weight and refuses to repair a class missing from an active product.
    """
    if not isinstance(base_lookup, CBClassWeightLookup):
        raise TypeError("base_lookup must be a prepared CBClassWeightLookup")
    if (
        base_lookup.field_names != pack.field_names
        or base_lookup.unknown_token != pack.unknown_token
    ):
        raise ValueError("base CB lookup does not match the active pack")
    if contract.fallback_weight != CLASS_WEIGHT_MAX:
        raise ValueError("CB zero-support fallback must equal the locked maximum")

    train = list(inputs.authoritative_train_skus)
    active = set(inputs.active_pool_skus)
    missing_observations: Counter[tuple[str, str]] = Counter()
    missing_products: dict[tuple[str, str], set[str]] = defaultdict(set)
    active_missing: set[tuple[str, str, str]] = set()
    for sku_id in train:
        row = inputs.rows_by_sku[sku_id]
        for field_name in pack.field_names:
            class_names = label_class_keys(
                field_name=field_name,
                label=row.labels[field_name],
                pack=pack,
            )
            for class_name in class_names:
                if class_name in base_lookup.weights[field_name]:
                    continue
                key = (field_name, class_name)
                missing_observations[key] += 1
                missing_products[key].add(sku_id)
                if sku_id in active:
                    active_missing.add((sku_id, field_name, class_name))

    if active_missing:
        example = sorted(active_missing)[0]
        raise ValueError(
            "base CB lookup is incomplete on the active pool: "
            f"{example[0]} {example[1]}/{example[2]}"
        )

    observed_entries = tuple(
        (field_name, class_name, observations)
        for (field_name, class_name), observations in sorted(
            missing_observations.items()
        )
    )
    if observed_entries != contract.expected_entries:
        raise ValueError("CB zero-active-support entry contract drifted")
    affected_products = set().union(*missing_products.values()) if missing_products else set()
    if len(affected_products) != contract.expected_affected_products:
        raise ValueError("CB zero-active-support product count drifted")

    extended_weights: dict[str, Mapping[str, float]] = {}
    for field_name in pack.field_names:
        field_weights = dict(base_lookup.weights[field_name])
        for observed_field, class_name, _ in observed_entries:
            if observed_field == field_name:
                if class_name in field_weights:
                    raise RuntimeError("CB extension would overwrite an active weight")
                field_weights[class_name] = contract.fallback_weight
        extended_weights[field_name] = MappingProxyType(field_weights)

    entry_ledger = [
        {
            "field_name": field_name,
            "class_name": class_name,
            "active_pool_support": 0,
            "full_training_observations": observations,
            "affected_products": len(missing_products[(field_name, class_name)]),
            "weight": contract.fallback_weight,
        }
        for field_name, class_name, observations in observed_entries
    ]
    entry_hash = hashlib.sha256(
        json.dumps(
            entry_ledger,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    audit = {
        "version": CB_DIAGNOSTIC_EXTENSION_VERSION,
        "role": "gold-only full-training diagnostic extension to active CB lookup",
        "base_artifact_sha256": CB_CLASS_WEIGHT_ARTIFACT_SHA256,
        "policy": {
            "trigger": "valid gold class has zero support in the active-pool map",
            "weight": contract.fallback_weight,
            "rationale": "limit of the existing rare-class formula after clipping",
            "active_weights_changed": False,
            "full_training_support_used_to_retune_weights": False,
        },
        "missing_attribute_class_pairs": len(observed_entries),
        "missing_class_observations": sum(
            observations for _, _, observations in observed_entries
        ),
        "affected_training_products": len(affected_products),
        "affected_active_products": 0,
        "ordered_entry_ledger_sha256": entry_hash,
        "entries": entry_ledger,
        "candidate_completion_rewards_calculated": False,
        "candidate_aggregates_calculated": False,
    }
    lookup = CBClassWeightLookup(
        artifact_version=(
            f"{base_lookup.artifact_version}+{CB_DIAGNOSTIC_EXTENSION_VERSION}"
        ),
        field_names=base_lookup.field_names,
        unknown_token=base_lookup.unknown_token,
        weights=MappingProxyType(extended_weights),
    )
    return FullTrainingCBExtension(
        base_lookup=base_lookup,
        lookup=lookup,
        audit=MappingProxyType(audit),
    )


def validate_cb_extension_for_publication(
    extension: FullTrainingCBExtension,
) -> dict[str, Any]:
    """Prove that the derived lookup is exactly base plus its audit ledger."""
    if not isinstance(extension, FullTrainingCBExtension):
        raise TypeError("cb_extension must be a FullTrainingCBExtension")
    if not isinstance(extension.base_lookup, CBClassWeightLookup) or not isinstance(
        extension.lookup, CBClassWeightLookup
    ):
        raise TypeError("CB extension lookups must be prepared CBClassWeightLookup values")
    base = extension.base_lookup
    derived = extension.lookup
    if (
        base.field_names != derived.field_names
        or base.unknown_token != derived.unknown_token
    ):
        raise ValueError("CB extension lookup does not match its base lookup")
    expected_artifact_version = (
        f"{base.artifact_version}+{CB_DIAGNOSTIC_EXTENSION_VERSION}"
    )
    if derived.artifact_version != expected_artifact_version:
        raise ValueError("CB extension lookup version drifted")

    # Canonical JSON conversion also rejects non-serializable or non-finite audit data.
    try:
        audit = json.loads(
            json.dumps(
                dict(extension.audit),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("CB extension audit is not canonical JSON data") from exc
    if audit.get("version") != CB_DIAGNOSTIC_EXTENSION_VERSION:
        raise ValueError("CB extension audit version drifted")
    if audit.get("role") != (
        "gold-only full-training diagnostic extension to active CB lookup"
    ):
        raise ValueError("CB extension audit role drifted")
    if audit.get("base_artifact_sha256") != CB_CLASS_WEIGHT_ARTIFACT_SHA256:
        raise ValueError("CB extension base artifact hash drifted")
    policy = audit.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("CB extension policy is missing")
    if policy.get("weight") != CLASS_WEIGHT_MAX:
        raise ValueError("CB extension policy weight drifted")
    if policy.get("active_weights_changed") is not False:
        raise ValueError("CB extension claims active weights changed")
    if policy.get("full_training_support_used_to_retune_weights") is not False:
        raise ValueError("CB extension claims full-training retuning")
    if audit.get("affected_active_products") != 0:
        raise ValueError("CB extension audit contains affected active products")
    if audit.get("candidate_completion_rewards_calculated") is not False:
        raise ValueError("CB extension audit used candidate completion rewards")
    if audit.get("candidate_aggregates_calculated") is not False:
        raise ValueError("CB extension audit used candidate aggregates")

    entries = audit.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("CB extension entry ledger is missing")
    expected_entry_keys: set[tuple[str, str]] = set()
    previous_key: tuple[str, str] | None = None
    observations = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("CB extension ledger entry is not an object")
        field_name = entry.get("field_name")
        class_name = entry.get("class_name")
        key = (field_name, class_name)
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("CB extension ledger contains an invalid class key")
        if previous_key is not None and key <= previous_key:
            raise ValueError("CB extension ledger is not uniquely ordered")
        previous_key = key
        expected_entry_keys.add(key)
        if entry.get("active_pool_support") != 0:
            raise ValueError("CB extension entry has nonzero active support")
        if entry.get("weight") != CLASS_WEIGHT_MAX:
            raise ValueError("CB extension entry weight drifted")
        entry_observations = entry.get("full_training_observations")
        affected_products = entry.get("affected_products")
        if not isinstance(entry_observations, int) or entry_observations <= 0:
            raise ValueError("CB extension entry has invalid observation count")
        if not isinstance(affected_products, int) or not 0 < affected_products <= entry_observations:
            raise ValueError("CB extension entry has invalid affected-product count")
        observations += entry_observations

    if audit.get("missing_attribute_class_pairs") != len(entries):
        raise ValueError("CB extension class-pair count drifted")
    if audit.get("missing_class_observations") != observations:
        raise ValueError("CB extension observation count drifted")
    affected_training_products = audit.get("affected_training_products")
    if not isinstance(affected_training_products, int) or affected_training_products <= 0:
        raise ValueError("CB extension affected-product total is invalid")
    entry_hash = hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if audit.get("ordered_entry_ledger_sha256") != entry_hash:
        raise ValueError("CB extension entry-ledger hash drifted")

    observed_additions: set[tuple[str, str]] = set()
    for field_name in base.field_names:
        base_weights = base.weights[field_name]
        derived_weights = derived.weights[field_name]
        for class_name, weight in base_weights.items():
            if derived_weights.get(class_name) != weight:
                raise ValueError("CB extension changed an active-pool weight")
        for class_name, weight in derived_weights.items():
            if class_name in base_weights:
                continue
            key = (field_name, class_name)
            observed_additions.add(key)
            if weight != CLASS_WEIGHT_MAX:
                raise ValueError("CB extension lookup addition weight drifted")
    if observed_additions != expected_entry_keys:
        raise ValueError("CB extension lookup additions differ from its ledger")
    return audit


def validate_full_training_scope(
    inputs: ReplayInputs,
    *,
    contract: FullTrainingScopeContract = PRODUCTION_SCOPE_CONTRACT,
) -> dict[str, Any]:
    """Prove membership, family separation, k=8 completeness and order."""
    if contract.role != FULL_TRAINING_ROLE:
        raise ValueError("full-training replay role drifted")
    train = list(inputs.authoritative_train_skus)
    validation = list(inputs.validation_skus)
    active = list(inputs.active_pool_skus)
    train_set = _require_unique("authoritative training scope", train)
    validation_set = _require_unique("SFT validation scope", validation)
    active_set = _require_unique("active training scope", active)

    if len(train) != contract.training_groups:
        raise ValueError("authoritative training group count drifted")
    if len(validation) != contract.validation_groups:
        raise ValueError("SFT validation group count drifted")
    if len(active) != contract.active_groups:
        raise ValueError("active training group count drifted")
    if train_set & validation_set:
        raise ValueError("authoritative training and SFT validation overlap")
    if not active_set <= train_set:
        raise ValueError("active pool is not a subset of authoritative training")

    row_skus = set(inputs.rows_by_sku)
    if train_set | validation_set != row_skus:
        raise ValueError("training and validation do not exactly cover source rows")
    if set(inputs.records_by_sku) != row_skus:
        raise ValueError("rollout groups do not exactly cover source rows")

    train_families = {group_key(inputs.rows_by_sku[sku]) for sku in train}
    validation_families = {
        group_key(inputs.rows_by_sku[sku]) for sku in validation
    }
    if train_families & validation_families:
        raise ValueError("training and validation product families overlap")

    expected_indices = list(range(ROLLOUTS_PER_PROMPT))
    all_keys: set[tuple[str, int]] = set()
    for sku_id in inputs.rows_by_sku:
        records = inputs.records_by_sku[sku_id]
        if len(records) != ROLLOUTS_PER_PROMPT:
            raise ValueError(f"{sku_id}: rollout group must contain exactly 8 records")
        indices = [record.rollout_index for record in records]
        if indices != expected_indices:
            raise ValueError(f"{sku_id}: rollout indices must be ordered 0 through 7")
        if any(record.sku_id != sku_id for record in records):
            raise ValueError(f"{sku_id}: rollout group contains a different SKU")
        for index in indices:
            key = (sku_id, index)
            if key in all_keys:
                raise ValueError(f"duplicate source rollout key: {key}")
            all_keys.add(key)

    ordered_sku_hash = ordered_sha256(train)
    ordered_key_hash = ordered_sha256(
        f"{sku_id}\t{index}"
        for sku_id in train
        for index in range(ROLLOUTS_PER_PROMPT)
    )
    if ordered_sku_hash != contract.ordered_sku_sha256:
        raise ValueError("authoritative training SKU order hash drifted")
    if ordered_key_hash != contract.ordered_rollout_key_sha256:
        raise ValueError("authoritative training rollout-key order hash drifted")

    return {
        "version": VERSION,
        "role": contract.role,
        "training_groups": len(train),
        "validation_groups_excluded": len(validation),
        "active_groups_included": len(active),
        "additional_training_groups": len(train_set - active_set),
        "completion_records": contract.completion_records,
        "ordered_sku_sha256": ordered_sku_hash,
        "ordered_rollout_key_sha256": ordered_key_hash,
        "training_validation_sku_overlap": 0,
        "training_validation_family_overlap": 0,
        "model_generation_performed": False,
        "candidate_aggregates_calculated": False,
    }


def iter_full_training_replay_groups(
    *,
    inputs: ReplayInputs,
    original_rewards: Mapping[tuple[str, int], Mapping[str, float]],
    pack: Pack,
    class_weights: CBClassWeightLookup,
    contract: FullTrainingScopeContract = PRODUCTION_SCOPE_CONTRACT,
) -> Iterable[dict[str, Any]]:
    """Yield manifest-ordered training groups through the shared scorer."""
    validate_full_training_scope(inputs, contract=contract)
    for position, sku_id in enumerate(inputs.authoritative_train_skus):
        yield build_replay_group(
            group_position=position,
            row=inputs.rows_by_sku[sku_id],
            records=inputs.records_by_sku[sku_id],
            original_rewards=original_rewards,
            pack=pack,
            class_weights=class_weights,
        )


def build_full_training_manifest(
    *,
    repo_root: str | Path,
    inputs: ReplayInputs,
    records_metadata: Mapping[str, Any],
    scope_validation: Mapping[str, Any],
    cb_extension: FullTrainingCBExtension,
    contract: FullTrainingScopeContract = PRODUCTION_SCOPE_CONTRACT,
    implementation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the raw-evidence manifest without aggregating candidate outcomes."""
    repo_root = Path(repo_root).resolve()
    implementation = Path(implementation_path or __file__).resolve()
    if scope_validation.get("role") != FULL_TRAINING_ROLE:
        raise ValueError("validated full-training role drifted")
    if records_metadata.get("jsonl_group_records") != contract.training_groups:
        raise ValueError("published full-training group count drifted")
    if records_metadata.get("completion_records") != contract.completion_records:
        raise ValueError("published full-training completion count drifted")
    if records_metadata.get("ordered_sku_sha256") != contract.ordered_sku_sha256:
        raise ValueError("published full-training SKU order hash drifted")
    if (
        records_metadata.get("ordered_rollout_key_sha256")
        != contract.ordered_rollout_key_sha256
    ):
        raise ValueError("published full-training rollout-key order hash drifted")
    extension_audit = validate_cb_extension_for_publication(cb_extension)

    return {
        "version": MANIFEST_VERSION,
        "status": "raw_evidence_published",
        "role": FULL_TRAINING_ROLE,
        "cuda_imports_performed": False,
        "selection_boundary": {
            "allowed": "all authoritative SFT-training products and locked k=8 rollouts",
            "excluded": f"all {contract.validation_groups} SFT-validation SKUs",
            "model_generation_performed": False,
            "validation_data_used": False,
            "legacy_frozen_300_used": False,
            "probe_100_used": False,
            "candidate_rewards_calculated": True,
            "aggregate_candidate_comparison_calculated": False,
            "candidate_rankings_calculated": False,
            "acceptance_thresholds_applied": False,
            "winner_selected": False,
        },
        "record_contract": {
            "grain": "one JSONL line per product group; eight ordered completions",
            "group_size": ROLLOUTS_PER_PROMPT,
            "candidate_order": list(CANDIDATES),
            "original_reward_included": True,
            "raw_completion_text_included": True,
            "candidate_component_ledgers_included": True,
            "cb_diagnostic_extension_included": True,
        },
        "inputs": _portable(inputs.metadata, repo_root=repo_root),
        "code": {
            "implementation": _file_metadata(implementation, repo_root=repo_root)
            if implementation.is_relative_to(repo_root)
            else {
                "path": str(implementation),
                "bytes": implementation.stat().st_size,
                "sha256": sha256_file(implementation),
            },
            "shared_group_builder": _file_metadata(
                Path(__file__).resolve().parent / "replay_run2_candidates.py",
                repo_root=repo_root,
            )
            if (Path(__file__).resolve().parent / "replay_run2_candidates.py").is_relative_to(repo_root)
            else {
                "path": str(Path(__file__).resolve().parent / "replay_run2_candidates.py"),
                "bytes": (Path(__file__).resolve().parent / "replay_run2_candidates.py").stat().st_size,
                "sha256": sha256_file(Path(__file__).resolve().parent / "replay_run2_candidates.py"),
            },
            "candidate_reward_version": REWARD_IMPLEMENTATION_VERSION,
            "cb_class_weight_artifact_sha256": CB_CLASS_WEIGHT_ARTIFACT_SHA256,
        },
        "cb_diagnostic_extension": extension_audit,
        "output": dict(records_metadata),
        "integrity": {
            **dict(scope_validation),
            "unique_rollout_keys": contract.completion_records,
            "all_groups_have_exactly_eight_ordered_completions": True,
            "same_source_completion_scored_by_all_candidates": True,
            "cb_extension_ledger_sha256": extension_audit[
                "ordered_entry_ledger_sha256"
            ],
            "cb_active_weights_changed": False,
        },
        "interpretation_guardrails": {
            "off_policy_return_estimate": False,
            "completion_independence_assumed": False,
            "candidate_superiority_claim_allowed": False,
            "gate_g10_result_available": False,
            "next_step": "analyze only after both replay scopes are published and verified",
        },
    }


def _publish_pair_exclusively(
    *,
    staged_records: Path,
    records_output: Path,
    staged_manifest: Path,
    manifest_output: Path,
) -> None:
    """Hard-link both staged files, rolling back only links created here."""
    published: list[Path] = []
    try:
        os.link(staged_records, records_output)
        published.append(records_output)
        os.link(staged_manifest, manifest_output)
        published.append(manifest_output)
    except BaseException:
        for path in reversed(published):
            path.unlink()
        raise


def publish_full_training_replay(
    *,
    repo_root: str | Path,
    inputs: ReplayInputs,
    original_rewards: Mapping[tuple[str, int], Mapping[str, float]],
    pack: Pack,
    cb_extension: FullTrainingCBExtension,
    records_output: str | Path = DEFAULT_RECORDS_OUTPUT,
    manifest_output: str | Path = DEFAULT_MANIFEST_OUTPUT,
    contract: FullTrainingScopeContract = PRODUCTION_SCOPE_CONTRACT,
) -> dict[str, Any]:
    """Stage, validate and exclusively publish a records/manifest pair."""
    repo_root = Path(repo_root).resolve()
    records_path = _resolve_within_repo(repo_root, records_output)
    manifest_path = _resolve_within_repo(repo_root, manifest_output)
    if not str(records_path).endswith(".jsonl.gz"):
        raise ValueError("full-training records path must end in .jsonl.gz")
    if manifest_path.suffix != ".json":
        raise ValueError("full-training manifest path must end in .json")
    if records_path == manifest_path:
        raise ValueError("full-training records and manifest paths must differ")

    active_records = _resolve_within_repo(repo_root, ACTIVE_RECORDS_OUTPUT)
    active_manifest = _resolve_within_repo(repo_root, ACTIVE_MANIFEST_OUTPUT)
    if records_path in {active_records, active_manifest} or manifest_path in {
        active_records,
        active_manifest,
    }:
        raise ValueError("full-training outputs cannot alias active replay outputs")
    collisions = [path for path in (records_path, manifest_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"full-training replay output already exists: {collisions}")

    records_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    scope_validation = validate_full_training_scope(inputs, contract=contract)
    extension_audit = validate_cb_extension_for_publication(cb_extension)

    with tempfile.TemporaryDirectory(
        prefix=".grpo-run2-full-training-replay.", dir=repo_root
    ) as staging_name:
        staging = Path(staging_name)
        staged_records = staging / "records.jsonl.gz"
        staged_manifest = staging / "manifest.json"
        groups = iter_full_training_replay_groups(
            inputs=inputs,
            original_rewards=original_rewards,
            pack=pack,
            class_weights=cb_extension.lookup,
            contract=contract,
        )
        staged_metadata = write_replay_groups(
            staged_records,
            groups,
            repo_root=repo_root,
        )
        records_metadata = {
            **staged_metadata,
            "path": _display_path(records_path, repo_root),
        }
        manifest = build_full_training_manifest(
            repo_root=repo_root,
            inputs=inputs,
            records_metadata=records_metadata,
            scope_validation=scope_validation,
            cb_extension=cb_extension,
            contract=contract,
        )
        if manifest["cb_diagnostic_extension"] != extension_audit:
            raise RuntimeError("CB extension manifest snapshot drifted")
        write_exclusive_atomic_json(staged_manifest, manifest)

        # Recheck after all expensive work to close the ordinary collision race.
        collisions = [path for path in (records_path, manifest_path) if path.exists()]
        if collisions:
            raise FileExistsError(
                f"full-training replay output appeared during staging: {collisions}"
            )
        _publish_pair_exclusively(
            staged_records=staged_records,
            records_output=records_path,
            staged_manifest=staged_manifest,
            manifest_output=manifest_path,
        )
    return manifest
