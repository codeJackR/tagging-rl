"""Strict k=5 frontier labeling state machine for Run 2 confirmation data.

Membership is an input, never a variable: every one of the 400 frozen products
owns five fixed perturbation slots. Failed, missing, malformed, schema-invalid,
or vocabulary-invalid attempts create a retry for that same slot. No product can
be dropped or replaced, and finalization is impossible until all 2,000 slots
contain one usable response with raw provider lineage retained.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from labeling.consensus import consensus_labels
from labeling.records import (
    AttributeLabel,
    Provenance,
    Row,
    RowInput,
    SelfConsistency,
    from_verifier_record,
)
from scripts.prelabel import (
    MAX_TOKENS,
    PERTURBATIONS,
    PROMPT_VERSION,
    build_system,
    build_user,
)
from training.audit_data_boundaries import sha256_file
from verifier import Pack, verify_record


VERSION = "grpo-run2-confirmation-labeling-state-v1"
BUNDLE_VERSION = "grpo-run2-confirmation-frontier-bundle-v1"
EXPECTED_PRODUCTS = 400
EXPECTED_SAMPLES_PER_PRODUCT = 5
EXPECTED_PROVIDER = "openai"
EXPECTED_MODEL = "gpt-5.6-luna"
FINAL_FILES = frozenset(
    {"attempts.jsonl", "frontier.jsonl", "labeling-state.json", "manifest.json"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_selected_rows(
    selected_rows: Sequence[Mapping[str, Any]], *, expected_products: int
) -> dict[str, Mapping[str, Any]]:
    if len(selected_rows) != expected_products:
        raise ValueError(
            f"selected membership has {len(selected_rows)} rows; expected {expected_products}"
        )
    by_sku: dict[str, Mapping[str, Any]] = {}
    forbidden = {"labels", "provenance", "difficulty", "prediction", "model_output"}
    for order, row in enumerate(selected_rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"selected row {order} is not an object")
        overlap = forbidden.intersection(row)
        if overlap:
            raise ValueError(f"selected row {order} has post-treatment fields: {sorted(overlap)}")
        sku = row.get("sku_id")
        if not isinstance(sku, str) or not sku:
            raise ValueError(f"selected row {order} has invalid sku_id")
        if sku in by_sku:
            raise ValueError(f"duplicate selected SKU: {sku}")
        RowInput.model_validate(row.get("input"))
        source = row.get("source")
        if not isinstance(source, str) or not source.startswith("shopify:"):
            raise ValueError(f"selected row {order} has invalid source")
        by_sku[sku] = row
    return by_sku


def _slot_custom_id(slot_order: int, variant: int, attempt: int) -> str:
    return f"c2-{slot_order:04d}-v{variant}-a{attempt}"


def _build_body(provider: Any, pack: Pack, row: Mapping[str, Any], variant: int) -> dict:
    schema = provider.adapt_schema(pack.json_schema())
    inp = RowInput.model_validate(row["input"])
    return provider.build_body(
        build_system(pack),
        build_user(inp, variant),
        schema,
        MAX_TOKENS,
    )


def build_initial_labeling_state(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    membership_identity: Mapping[str, Any],
    pack: Pack,
    provider: Any,
    expected_products: int = EXPECTED_PRODUCTS,
    samples_per_product: int = EXPECTED_SAMPLES_PER_PRODUCT,
) -> dict[str, Any]:
    """Create the exact 400 x 5 logical request ledger before submission."""

    _validate_selected_rows(selected_rows, expected_products=expected_products)
    if samples_per_product != len(PERTURBATIONS) or samples_per_product != 5:
        raise ValueError("confirmation labeling requires all five locked perturbations")
    if provider.name != EXPECTED_PROVIDER or provider.model != EXPECTED_MODEL:
        raise ValueError(
            f"confirmation provider/model must be {EXPECTED_PROVIDER}/{EXPECTED_MODEL}"
        )
    if pack.category_field is None or len(pack.field_names) != 15:
        raise ValueError("confirmation pack must contain the locked 15 fields")

    slots = []
    slot_order = 0
    for membership_order, row in enumerate(selected_rows, start=1):
        for variant in range(samples_per_product):
            slot_order += 1
            body = _build_body(provider, pack, row, variant)
            slots.append(
                {
                    "slot_order": slot_order,
                    "membership_order": membership_order,
                    "sku_id": row["sku_id"],
                    "variant": variant,
                    "perturbation": PERTURBATIONS[variant],
                    "request_body_sha256": _sha256_json(body),
                    "attempts": [],
                    "pending": {
                        "attempt": 1,
                        "custom_id": _slot_custom_id(slot_order, variant, 1),
                    },
                    "usable_attempt": None,
                }
            )
    return {
        "version": VERSION,
        "status": "ready_for_initial_submission",
        "membership_identity": dict(membership_identity),
        "configuration": {
            "provider": provider.name,
            "model": provider.model,
            "prompt_version": PROMPT_VERSION,
            "samples_per_product": samples_per_product,
            "perturbation_count": len(PERTURBATIONS),
            "structured_outputs": True,
            "maximum_tokens": MAX_TOKENS,
            "consensus": "labeling.consensus.consensus_labels",
            "failed_request_policy": "retry same SKU and variant; never replace membership",
        },
        "counts": {
            "products": expected_products,
            "logical_slots": expected_products * samples_per_product,
            "usable_slots": 0,
            "pending_slots": expected_products * samples_per_product,
            "attempts": 0,
            "failed_attempts": 0,
            "batches": 0,
        },
        "batches": [],
        "slots": slots,
        "execution_boundary": {
            "membership_changed": False,
            "human_review_performed": False,
            "sft_or_grpo_model_predictions_generated": False,
            "gpu_training_performed": False,
        },
    }


def build_pending_submission_items(
    *,
    state: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    pack: Pack,
    provider: Any,
) -> list[tuple[str, dict[str, Any]]]:
    """Rebuild pending request bodies and prove they match the predeclared hashes."""

    if state.get("version") != VERSION:
        raise ValueError("unexpected confirmation labeling-state version")
    by_sku = _validate_selected_rows(
        selected_rows, expected_products=state["counts"]["products"]
    )
    if provider.name != state["configuration"]["provider"]:
        raise ValueError("provider drifted from labeling state")
    if provider.model != state["configuration"]["model"]:
        raise ValueError("model drifted from labeling state")

    items: list[tuple[str, dict[str, Any]]] = []
    for slot in state["slots"]:
        pending = slot["pending"]
        if pending is None:
            continue
        body = _build_body(provider, pack, by_sku[slot["sku_id"]], slot["variant"])
        if _sha256_json(body) != slot["request_body_sha256"]:
            raise ValueError(f"request body drifted for slot {slot['slot_order']}")
        items.append((pending["custom_id"], body))
    if len({custom_id for custom_id, _body in items}) != len(items):
        raise ValueError("pending submission contains duplicate custom IDs")
    return items


def _usable_output(text: str | None, error: str | None, pack: Pack) -> dict[str, Any]:
    if error:
        return {"usable": False, "failure": error, "parsed": None, "labels": None}
    if not text:
        return {"usable": False, "failure": "empty response text", "parsed": None, "labels": None}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "usable": False,
            "failure": f"JSONDecodeError: {exc.msg}",
            "parsed": None,
            "labels": None,
        }
    if not isinstance(parsed, dict):
        return {"usable": False, "failure": "response JSON was not an object", "parsed": None, "labels": None}
    verified = verify_record(parsed, pack)
    if not verified.schema_valid or not verified.vocab_valid:
        return {
            "usable": False,
            "failure": "; ".join(verified.errors)[:500] or "schema/vocabulary invalid",
            "parsed": parsed,
            "labels": None,
        }
    labels = from_verifier_record(parsed, pack)
    if set(labels) != set(pack.field_names) or len(labels) != 15:
        return {"usable": False, "failure": "response did not contain all 15 fields", "parsed": parsed, "labels": None}
    return {
        "usable": True,
        "failure": None,
        "parsed": parsed,
        "labels": {
            name: label.model_dump(mode="json") for name, label in labels.items()
        },
        "rule_violations": verified.rule_violations,
    }


def apply_batch_results(
    *,
    state: Mapping[str, Any],
    batch_id: str,
    results: Sequence[Mapping[str, Any]],
    pack: Pack,
) -> dict[str, Any]:
    """Apply one batch and schedule only same-slot retries for unusable results."""

    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id must be non-empty")
    updated = copy.deepcopy(state)
    if batch_id in {batch["batch_id"] for batch in updated["batches"]}:
        raise ValueError(f"batch already applied: {batch_id}")
    pending_by_id = {
        slot["pending"]["custom_id"]: slot
        for slot in updated["slots"]
        if slot["pending"] is not None
    }
    result_by_id: dict[str, Mapping[str, Any]] = {}
    for result in results:
        custom_id = result.get("custom_id")
        if custom_id not in pending_by_id:
            raise ValueError(f"unexpected result custom_id: {custom_id!r}")
        if custom_id in result_by_id:
            raise ValueError(f"duplicate result custom_id: {custom_id}")
        result_by_id[str(custom_id)] = result

    usable_in_batch = 0
    failed_in_batch = 0
    for custom_id, slot in pending_by_id.items():
        pending = slot["pending"]
        result = result_by_id.get(custom_id)
        if result is None:
            result = {"custom_id": custom_id, "text": None, "error": "missing batch result"}
        evaluation = _usable_output(result.get("text"), result.get("error"), pack)
        attempt_record = {
            "attempt": pending["attempt"],
            "custom_id": custom_id,
            "batch_id": batch_id,
            "provider_result_id": result.get("provider_result_id"),
            "provider_request_id": result.get("provider_request_id"),
            "usage": result.get("usage"),
            "raw_text": result.get("text"),
            "provider_error": result.get("error"),
            **evaluation,
        }
        slot["attempts"].append(attempt_record)
        if evaluation["usable"]:
            slot["usable_attempt"] = pending["attempt"]
            slot["pending"] = None
            usable_in_batch += 1
        else:
            next_attempt = pending["attempt"] + 1
            slot["pending"] = {
                "attempt": next_attempt,
                "custom_id": _slot_custom_id(
                    slot["slot_order"], slot["variant"], next_attempt
                ),
            }
            failed_in_batch += 1

    updated["batches"].append(
        {
            "batch_order": len(updated["batches"]) + 1,
            "batch_id": batch_id,
            "expected_results": len(pending_by_id),
            "received_results": len(results),
            "usable_results": usable_in_batch,
            "failed_or_missing_results": failed_in_batch,
            "custom_ids": sorted(pending_by_id),
        }
    )
    slots = updated["slots"]
    usable = sum(slot["usable_attempt"] is not None for slot in slots)
    attempts = sum(len(slot["attempts"]) for slot in slots)
    failures = sum(
        not attempt["usable"] for slot in slots for attempt in slot["attempts"]
    )
    updated["counts"].update(
        {
            "usable_slots": usable,
            "pending_slots": len(slots) - usable,
            "attempts": attempts,
            "failed_attempts": failures,
            "batches": len(updated["batches"]),
        }
    )
    updated["status"] = (
        "all_slots_usable_ready_to_finalize"
        if usable == len(slots)
        else "retry_required_for_same_membership"
    )
    return updated


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, values: Sequence[Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(_canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _identity(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finalize_frontier_bundle(
    *,
    output_dir: str | Path,
    state: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    pack: Pack,
) -> dict[str, Any]:
    """Atomically publish raw attempts plus 400 k=5 consensus rows."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"frontier output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"frontier output parent does not exist: {output_dir.parent}")
    if state.get("status") != "all_slots_usable_ready_to_finalize":
        raise ValueError("frontier state still has unusable or pending slots")
    products = state["counts"]["products"]
    k = state["configuration"]["samples_per_product"]
    by_sku = _validate_selected_rows(selected_rows, expected_products=products)
    if state["counts"]["usable_slots"] != products * k:
        raise ValueError("usable slot count does not equal products x k")

    attempts = []
    samples_by_sku: dict[str, list[dict]] = {sku: [] for sku in by_sku}
    for slot in state["slots"]:
        usable_attempts = [attempt for attempt in slot["attempts"] if attempt["usable"]]
        if len(usable_attempts) != 1:
            raise ValueError(f"slot {slot['slot_order']} does not have exactly one usable attempt")
        usable = usable_attempts[0]
        samples_by_sku[slot["sku_id"]].append(usable["labels"])
        for attempt in slot["attempts"]:
            attempts.append(
                {
                    "slot_order": slot["slot_order"],
                    "membership_order": slot["membership_order"],
                    "sku_id": slot["sku_id"],
                    "variant": slot["variant"],
                    "perturbation": slot["perturbation"],
                    **attempt,
                }
            )

    frontier_rows = []
    agreement_values = []
    for membership_order, row in enumerate(selected_rows, start=1):
        sku = str(row["sku_id"])
        sample_models = [
            {
                name: AttributeLabel.model_validate(label)
                for name, label in sample.items()
            }
            for sample in samples_by_sku[sku]
        ]
        if len(sample_models) != k:
            raise ValueError(f"SKU {sku} has {len(sample_models)} usable samples; expected {k}")
        labels, agreement = consensus_labels(sample_models)
        agreement_values.extend(agreement.values())
        frontier_rows.append(
            Row(
                sku_id=sku,
                source=str(row["source"]),
                split="eval",
                input=RowInput.model_validate(row["input"]),
                labels=labels,
                provenance=Provenance(
                    labeler=f"{state['configuration']['model']}@{PROMPT_VERSION}",
                    prompt_version=PROMPT_VERSION,
                    self_consistency=SelfConsistency(k=k, agreement=agreement),
                    frontier_labels=copy.deepcopy(labels),
                ),
            ).model_dump(mode="json")
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    ).resolve()
    try:
        attempts_path = staging / "attempts.jsonl"
        frontier_path = staging / "frontier.jsonl"
        state_path = staging / "labeling-state.json"
        manifest_path = staging / "manifest.json"
        _write_jsonl(attempts_path, attempts)
        _write_jsonl(frontier_path, frontier_rows)
        _write_json(state_path, state)
        manifest = {
            "version": BUNDLE_VERSION,
            "status": "frontier_labels_complete_awaiting_human_review",
            "membership_identity": dict(state["membership_identity"]),
            "configuration": dict(state["configuration"]),
            "counts": {
                "products": products,
                "attributes_per_product": len(pack.field_names),
                "logical_slots": products * k,
                "usable_slots": state["counts"]["usable_slots"],
                "attempts": len(attempts),
                "failed_attempts": state["counts"]["failed_attempts"],
                "batches": state["counts"]["batches"],
                "frontier_cells": products * len(pack.field_names),
            },
            "agreement": {
                "mean_cell_agreement": sum(agreement_values) / len(agreement_values),
                "unanimous_cells": sum(value == 1.0 for value in agreement_values),
                "nonunanimous_cells": sum(value < 1.0 for value in agreement_values),
            },
            "files": {
                "attempts": _identity(attempts_path),
                "frontier": _identity(frontier_path),
                "labeling_state": _identity(state_path),
            },
            "invariants": {
                "membership_changed": False,
                "exactly_five_usable_samples_per_product": True,
                "failed_slots_retried_without_product_replacement": True,
                "all_attempts_and_raw_text_retained": True,
                "human_review_complete": False,
                "published_exclusively_and_atomically": True,
            },
            "execution_boundary": {
                "human_review_performed": False,
                "sft_or_grpo_model_predictions_generated": False,
                "gpu_training_performed": False,
            },
        }
        _write_json(manifest_path, manifest)
        if {path.name for path in staging.iterdir()} != FINAL_FILES:
            raise RuntimeError("frontier staging bundle has unexpected files")
        os.rename(staging, output_dir)
        if staging.exists() or not output_dir.is_dir():
            raise RuntimeError("atomic frontier publication did not complete")
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
