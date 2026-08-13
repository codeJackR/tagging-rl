"""Final collision-protected freeze for the untouched Run 2 confirmation set.

This is the last pre-inference boundary. It verifies reviewed rows against the
pre-label selection order, prior SKU/family exclusions, frontier/review lineage,
and the locked pack; publishes only data and audit metadata; and creates a
successor role manifest that preserves the original "confirmation required"
manifest as immutable history.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from labeling.records import Row, canonical_line
from training.audit_data_boundaries import (
    sha256_file,
    sku_set_sha256,
    write_exclusive_atomic_json,
)
from training.split_sft import group_key
from verifier import Pack, verify_record


VERSION = "grpo-run2-confirmation-freeze-v1"
ROLE_VERSION = "grpo-run2-data-role-manifest-v2"
EXPECTED_PRODUCTS = 400
EXPECTED_OUTPUT_NAME = "confirmation_run2_v1"
EXPECTED_ROLE_OUTPUT_NAME = "grpo-run2-data-role-manifest-confirmation-assigned.json"
FINAL_FILES = frozenset({"eval.jsonl", "manifest.json"})
REQUIRED_LINEAGE = frozenset(
    {
        "source_terms_audit",
        "acquisition_manifest",
        "selection_manifest",
        "frontier_labeling_manifest",
        "review_manifest",
        "reviewed_dataset",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _validate_identity(name: str, identity: Mapping[str, Any]) -> None:
    path = identity.get("path")
    digest = identity.get("sha256")
    size = identity.get("bytes")
    if not isinstance(path, str) or not path:
        raise ValueError(f"lineage {name} has no path")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ValueError(f"lineage {name} has invalid SHA-256")
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"lineage {name} has invalid byte count")


def _validate_lineage(lineage: Mapping[str, Mapping[str, Any]]) -> None:
    if set(lineage) != REQUIRED_LINEAGE:
        raise ValueError(
            f"confirmation freeze lineage drifted: {sorted(lineage)} != {sorted(REQUIRED_LINEAGE)}"
        )
    for name, identity in lineage.items():
        _validate_identity(name, identity)


def _validate_review_manifest(review: Mapping[str, Any]) -> None:
    if review.get("status") != "human_review_complete_ready_for_final_freeze":
        raise ValueError("review bundle is not ready for final freeze")
    counts = review.get("counts", {})
    required = {
        "products": EXPECTED_PRODUCTS,
        "primary_reviewed_cells": 6_000,
        "second_reviewed_products": 40,
        "second_reviewed_cells": 600,
        "unresolved_cells": 0,
    }
    for key, expected in required.items():
        if counts.get(key) != expected:
            raise ValueError(f"review manifest {key}={counts.get(key)!r}, expected {expected}")
    invariants = review.get("invariants", {})
    for key in (
        "all_6000_primary_cells_reviewed",
        "all_600_second_review_cells_reviewed",
        "second_review_independent",
        "all_disagreements_adjudicated",
        "all_final_rows_schema_vocab_rule_valid",
        "support_shortfalls_disclosed_without_membership_changes",
    ):
        if invariants.get(key) is not True:
            raise ValueError(f"review invariant is not true: {key}")


def _validate_selection(selection: Mapping[str, Any]) -> list[str]:
    if selection.get("status") != "confirmation_membership_selected_before_labeling":
        raise ValueError("selection manifest is not a frozen pre-label membership")
    selected = selection.get("selected")
    if not isinstance(selected, list) or len(selected) != EXPECTED_PRODUCTS:
        raise ValueError("selection manifest does not contain exactly 400 rows")
    ordered = [item.get("sku_id") for item in selected]
    if any(not isinstance(sku, str) or not sku for sku in ordered):
        raise ValueError("selection manifest contains invalid SKU IDs")
    if len(set(ordered)) != EXPECTED_PRODUCTS:
        raise ValueError("selection manifest contains duplicate SKU IDs")
    if selection.get("invariants", {}).get("membership_uses_labels_or_model_outputs") is not False:
        raise ValueError("selection membership boundary is not outcome-free")
    return ordered


def validate_confirmation_for_freeze(
    *,
    reviewed_rows: Sequence[Row | Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
    review_manifest: Mapping[str, Any],
    source_gate_result: Mapping[str, Any],
    prior_sku_ids: set[str] | frozenset[str],
    prior_family_keys: set[str] | frozenset[str],
    pack: Pack,
) -> tuple[list[Row], dict[str, Any]]:
    """Prove all membership, review, overlap and verifier invariants."""

    if source_gate_result.get("passed") is not True:
        raise ValueError("source permission gate did not pass")
    if source_gate_result.get("approved_store_count", 0) < 8:
        raise ValueError("source gate has fewer than eight approved stores")
    if len(prior_sku_ids) != 4_000:
        raise ValueError(f"prior exclusion universe has {len(prior_sku_ids)} SKUs, expected 4000")
    _validate_review_manifest(review_manifest)
    selected_order = _validate_selection(selection_manifest)
    rows = [row if isinstance(row, Row) else Row.model_validate(row) for row in reviewed_rows]
    if len(rows) != EXPECTED_PRODUCTS:
        raise ValueError(f"reviewed dataset has {len(rows)} rows, expected 400")
    row_order = [row.sku_id for row in rows]
    if row_order != selected_order:
        raise ValueError("reviewed dataset order or membership differs from pre-label selection")

    exact_overlap = sorted(set(row_order).intersection(prior_sku_ids))
    family_keys = [group_key(row) for row in rows]
    family_overlap = sorted(set(family_keys).intersection(prior_family_keys))
    if exact_overlap:
        raise ValueError(f"confirmation has prior exact-SKU overlap: {exact_overlap[:3]}")
    if family_overlap:
        raise ValueError(f"confirmation has prior family overlap: {family_overlap[:3]}")

    corrected_rows = 0
    for row in rows:
        if row.split != "eval":
            raise ValueError(f"confirmation SKU {row.sku_id} is not split=eval")
        if set(row.labels) != set(pack.field_names):
            raise ValueError(f"confirmation SKU {row.sku_id} lacks all 15 labels")
        if row.provenance.frontier_labels is None:
            raise ValueError(f"confirmation SKU {row.sku_id} lacks frontier snapshot")
        if (
            row.provenance.self_consistency is None
            or row.provenance.self_consistency.k != 5
        ):
            raise ValueError(f"confirmation SKU {row.sku_id} lacks k=5 provenance")
        verified = verify_record(row.to_verifier_record(pack), pack)
        if not verified.schema_valid or not verified.vocab_valid or verified.rule_violations:
            raise ValueError(
                f"confirmation SKU {row.sku_id} failed final verifier: "
                f"errors={verified.errors[:3]} rules={verified.rule_violations[:3]}"
            )
        corrected_rows += row.provenance.human_corrected

    summary = {
        "products": len(rows),
        "attributes_per_product": len(pack.field_names),
        "cells": len(rows) * len(pack.field_names),
        "stores": len({row.source for row in rows}),
        "families": len(set(family_keys)),
        "human_corrected_rows": corrected_rows,
        "prior_exact_sku_overlap": 0,
        "prior_family_overlap": 0,
        "ordered_sku_sha256": sku_set_sha256(row_order),
        "ordered_family_sha256": sku_set_sha256(family_keys),
    }
    return rows, summary


def build_assigned_role_manifest(
    *,
    parent: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    freeze_manifest_identity: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    freeze_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a v2 role decision without overwriting the v1 historical state."""

    if parent.get("status") != "development_roles_locked_confirmation_required":
        raise ValueError("parent data-role manifest does not require confirmation")
    if parent.get("final_confirmation", {}).get("assigned") is not False:
        raise ValueError("parent data-role manifest already assigns confirmation")
    _validate_identity("parent_data_role_manifest", parent_identity)
    _validate_identity("freeze_manifest", freeze_manifest_identity)
    _validate_identity("confirmation_dataset", dataset_identity)

    role = copy.deepcopy(parent)
    role["version"] = ROLE_VERSION
    role["status"] = "development_and_confirmation_roles_locked"
    role["parent_manifest"] = dict(parent_identity)
    role["dataset_dispositions"]["new_untouched_confirmation"] = (
        "assigned_final_confirmation_sealed_until_recipe_lock"
    )
    role["final_confirmation"] = {
        **role["final_confirmation"],
        "assigned": True,
        "dataset": dict(dataset_identity),
        "freeze_manifest": dict(freeze_manifest_identity),
        "rows": freeze_summary["products"],
        "families": freeze_summary["families"],
        "stores": freeze_summary["stores"],
        "exact_sku_overlap_with_prior_4000": freeze_summary["prior_exact_sku_overlap"],
        "normalized_family_overlap_with_prior_4000": freeze_summary["prior_family_overlap"],
        "labels_opened_for_run2_selection": False,
        "model_outputs_generated": False,
        "aggregate_confirmation_metrics_calculated": False,
        "allowed_next_use": "one final comparison after recipe and checkpoint lock",
    }
    role["phase_e_gate"] = {
        **role["phase_e_gate"],
        "blocking_reason": None,
        "final_confirmation_assigned": True,
        "passed": True,
    }
    role["execution_boundary"] = {
        **role["execution_boundary"],
        "model_inference_performed": False,
        "confirmation_metrics_calculated": False,
        "run2_training_contract_locked": False,
    }
    role.setdefault("inputs", {})["confirmation_dataset"] = dict(dataset_identity)
    role["inputs"]["confirmation_freeze_manifest"] = dict(freeze_manifest_identity)
    return role


def _write_eval(path: Path, rows: Sequence[Row]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_line(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def freeze_confirmation_bundle(
    *,
    output_dir: str | Path,
    role_output: str | Path,
    reviewed_rows: Sequence[Row | Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
    review_manifest: Mapping[str, Any],
    support_report: Mapping[str, Any],
    source_gate_result: Mapping[str, Any],
    prior_sku_ids: set[str] | frozenset[str],
    prior_family_keys: set[str] | frozenset[str],
    pack: Pack,
    lineage: Mapping[str, Mapping[str, Any]],
    parent_role_manifest: Mapping[str, Any],
    parent_role_identity: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    frozen_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish dataset first, then its non-destructive role-manifest successor."""

    output_dir = Path(output_dir).resolve()
    role_output = Path(role_output).resolve()
    if output_dir.name != EXPECTED_OUTPUT_NAME:
        raise ValueError(f"confirmation output directory must be named {EXPECTED_OUTPUT_NAME}")
    if role_output.name != EXPECTED_ROLE_OUTPUT_NAME:
        raise ValueError(f"role output must be named {EXPECTED_ROLE_OUTPUT_NAME}")
    if output_dir.exists():
        raise FileExistsError(f"confirmation output already exists: {output_dir}")
    if role_output.exists():
        raise FileExistsError(f"confirmation role output already exists: {role_output}")
    if not output_dir.parent.is_dir() or not role_output.parent.is_dir():
        raise FileNotFoundError("confirmation output parents must already exist")
    _validate_lineage(lineage)
    rows, summary = validate_confirmation_for_freeze(
        reviewed_rows=reviewed_rows,
        selection_manifest=selection_manifest,
        review_manifest=review_manifest,
        source_gate_result=source_gate_result,
        prior_sku_ids=prior_sku_ids,
        prior_family_keys=prior_family_keys,
        pack=pack,
    )
    if support_report.get("target_per_attribute_status_or_value") != 8:
        raise ValueError("support report does not use the locked target of eight")
    if support_report.get("shortfalls_change_membership") is not False:
        raise ValueError("support report allows post-label membership changes")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    ).resolve()
    try:
        eval_path = staging / "eval.jsonl"
        manifest_path = staging / "manifest.json"
        _write_eval(eval_path, rows)
        dataset_identity = _identity(eval_path)
        manifest = {
            "version": VERSION,
            "status": "confirmation_frozen_sealed_before_final_recipe_lock",
            "role": "untouched_final_confirmation",
            "frozen_at_utc": frozen_at_utc,
            "dataset": dataset_identity,
            "exact_row_order": [row.sku_id for row in rows],
            "summary": summary,
            "lineage": {name: dict(identity) for name, identity in sorted(lineage.items())},
            "pack": {
                "name": pack.name,
                "vocab_sha256": sha256_file(pack.path / "vocab.yaml"),
                "rules_sha256": sha256_file(pack.path / "rules.yaml"),
            },
            "review": {
                "primary_reviewed_cells": review_manifest["counts"]["primary_reviewed_cells"],
                "second_reviewed_cells": review_manifest["counts"]["second_reviewed_cells"],
                "agreement_before_adjudication": review_manifest["agreement_before_adjudication"],
                "adjudicated_cells": review_manifest["counts"]["adjudicated_cells"],
                "unresolved_cells": 0,
                "support_shortfalls": len(support_report.get("shortfalls", [])),
            },
            "code": dict(code_identity),
            "role_successor_path": role_output.name,
            "decision_boundary": {
                "labels_used_for_recipe_or_checkpoint_selection": False,
                "sft_or_grpo_predictions_generated": False,
                "aggregate_confirmation_metrics_calculated": False,
                "membership_can_change_after_freeze": False,
                "next_allowed_action": "seal until one final recipe and checkpoint are locked",
            },
            "invariants": {
                "exactly_400_products": True,
                "exactly_6000_reviewed_cells": True,
                "zero_prior_sku_overlap": True,
                "zero_prior_family_overlap": True,
                "all_rows_schema_vocab_rule_valid": True,
                "frontier_and_human_provenance_retained": True,
                "support_shortfalls_disclosed_without_resampling": True,
                "published_exclusively_and_atomically": True,
            },
        }
        _write_json(manifest_path, manifest)
        if {path.name for path in staging.iterdir()} != FINAL_FILES:
            raise RuntimeError("confirmation freeze staging has unexpected files")
        os.rename(staging, output_dir)
        if staging.exists() or not output_dir.is_dir():
            raise RuntimeError("atomic confirmation freeze did not complete")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    freeze_manifest_identity = {
        "path": f"{output_dir.name}/manifest.json",
        "bytes": (output_dir / "manifest.json").stat().st_size,
        "sha256": sha256_file(output_dir / "manifest.json"),
    }
    dataset_role_identity = {
        "path": f"{output_dir.name}/eval.jsonl",
        "bytes": (output_dir / "eval.jsonl").stat().st_size,
        "sha256": sha256_file(output_dir / "eval.jsonl"),
    }
    role = build_assigned_role_manifest(
        parent=parent_role_manifest,
        parent_identity=parent_role_identity,
        freeze_manifest_identity=freeze_manifest_identity,
        dataset_identity=dataset_role_identity,
        freeze_summary=summary,
    )
    write_exclusive_atomic_json(role_output, role)
    return manifest, role
