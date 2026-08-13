"""Complete, blinded human-review boundary for Run 2 confirmation labels.

The primary packet contains all 400 x 15 cells.  A separately generated packet
contains all 15 cells for the deterministic 40-product second-review sample and
never contains primary decisions.  Imports require an explicit human decision
for every cell, preserve every correction, enforce reviewer independence, and
block final publication until every disagreement is adjudicated and all 400
final records pass schema, vocabulary, and rule verification.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from labeling.records import AttributeLabel, LabelStatus, Row
from training.audit_data_boundaries import sha256_file
from verifier import Pack, verify_record


VERSION = "grpo-run2-confirmation-review-v1"
PACKET_BUNDLE_VERSION = "grpo-run2-confirmation-review-packets-v1"
FINAL_BUNDLE_VERSION = "grpo-run2-confirmation-reviewed-bundle-v1"
EXPECTED_PRODUCTS = 400
ATTRIBUTES_PER_PRODUCT = 15
PRIMARY_CELLS = 6_000
SECOND_REVIEW_PRODUCTS = 40
SECOND_REVIEW_CELLS = 600
SECOND_REVIEW_SEED = "20260813-review"
SUPPORT_TARGET = 8

IMMUTABLE_COLUMNS = (
    "cell_id",
    "membership_order",
    "sku_id",
    "source",
    "title",
    "description",
    "brand",
    "category",
    "raw_tags_json",
    "image_url",
    "attribute",
    "field_kind",
    "allowed_values_json",
    "applies_to_json",
    "frontier_status",
    "frontier_value_json",
    "frontier_agreement",
    "sample_labels_json",
    "frontier_rule_violations_json",
)
MUTABLE_COLUMNS = (
    "reviewer_id",
    "decision",
    "corrected_status",
    "corrected_value_json",
    "rationale",
    "reviewed_at_utc",
)
PACKET_COLUMNS = IMMUTABLE_COLUMNS + MUTABLE_COLUMNS
PACKET_FILES = frozenset(
    {"manifest.json", "primary-review.csv", "secondary-review.csv"}
)
FINAL_FILES = frozenset(
    {"decisions.jsonl", "manifest.json", "reviewed.jsonl", "support.json"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SECOND_REVIEW_SEED}\0{value}".encode()).hexdigest()


def _rows(
    frontier_rows: Sequence[Row | Mapping[str, Any]], *, expected_products: int
) -> list[Row]:
    rows = [row if isinstance(row, Row) else Row.model_validate(row) for row in frontier_rows]
    if len(rows) != expected_products:
        raise ValueError(f"frontier has {len(rows)} rows; expected {expected_products}")
    skus = [row.sku_id for row in rows]
    if len(set(skus)) != len(skus):
        raise ValueError("frontier contains duplicate SKU IDs")
    return rows


def select_second_review_skus(
    frontier_rows: Sequence[Row | Mapping[str, Any]],
    *,
    expected_products: int = EXPECTED_PRODUCTS,
    audit_products: int = SECOND_REVIEW_PRODUCTS,
) -> list[str]:
    rows = _rows(frontier_rows, expected_products=expected_products)
    if audit_products <= 0 or audit_products > len(rows):
        raise ValueError("audit_products must be within the frontier size")
    return [
        row.sku_id
        for row in sorted(rows, key=lambda item: (_stable_hash(item.sku_id), item.sku_id))[
            :audit_products
        ]
    ]


def _attempt_samples(
    attempts: Sequence[Mapping[str, Any]], pack: Pack
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    samples: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for attempt in attempts:
        if not attempt.get("usable"):
            continue
        sku = attempt.get("sku_id")
        labels = attempt.get("labels")
        if not isinstance(sku, str) or not isinstance(labels, Mapping):
            raise ValueError("usable attempt lacks SKU or parsed labels")
        if set(labels) != set(pack.field_names):
            raise ValueError(f"usable attempt for {sku} lacks all pack fields")
        for attribute in pack.field_names:
            label = AttributeLabel.model_validate(labels[attribute])
            samples.setdefault((sku, attribute), []).append(label.model_dump(mode="json"))
    return samples


def build_review_packets(
    *,
    frontier_rows: Sequence[Row | Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    pack: Pack,
    expected_products: int = EXPECTED_PRODUCTS,
    audit_products: int = SECOND_REVIEW_PRODUCTS,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Build primary and blinded secondary packets directly from frontier data."""

    rows = _rows(frontier_rows, expected_products=expected_products)
    if len(pack.field_names) != ATTRIBUTES_PER_PRODUCT:
        raise ValueError("review pack must contain exactly 15 attributes")
    second_skus = select_second_review_skus(
        rows, expected_products=expected_products, audit_products=audit_products
    )
    second_set = set(second_skus)
    samples = _attempt_samples(attempts, pack)
    primary: list[dict[str, str]] = []
    secondary: list[dict[str, str]] = []

    for membership_order, row in enumerate(rows, start=1):
        record = row.to_verifier_record(pack)
        verified = verify_record(record, pack)
        for attribute in pack.field_names:
            label = row.labels.get(attribute)
            if label is None:
                raise ValueError(f"frontier SKU {row.sku_id} is missing {attribute}")
            spec = pack.specs[attribute]
            sample_labels = samples.get((row.sku_id, attribute), [])
            expected_k = row.provenance.self_consistency.k if row.provenance.self_consistency else 0
            if len(sample_labels) != expected_k or expected_k != 5:
                raise ValueError(
                    f"frontier SKU {row.sku_id} {attribute} has "
                    f"{len(sample_labels)} raw samples; expected 5"
                )
            base = {
                "cell_id": f"{membership_order:04d}::{attribute}",
                "membership_order": str(membership_order),
                "sku_id": row.sku_id,
                "source": row.source,
                "title": row.input.title,
                "description": row.input.description,
                "brand": row.input.brand or "",
                "category": row.input.category or "",
                "raw_tags_json": _canonical_json(row.input.raw_tags),
                "image_url": row.input.image_url or "",
                "attribute": attribute,
                "field_kind": spec.kind,
                "allowed_values_json": _canonical_json(list(spec.values)),
                "applies_to_json": _canonical_json(
                    sorted(spec.applies_to) if spec.applies_to is not None else None
                ),
                "frontier_status": label.status.value,
                "frontier_value_json": _canonical_json(label.value),
                "frontier_agreement": str(
                    row.provenance.self_consistency.agreement.get(attribute, 0.0)
                    if row.provenance.self_consistency
                    else 0.0
                ),
                "sample_labels_json": _canonical_json(sample_labels),
                "frontier_rule_violations_json": _canonical_json(
                    verified.rule_violations
                ),
                "reviewer_id": "",
                "decision": "",
                "corrected_status": "",
                "corrected_value_json": "",
                "rationale": "",
                "reviewed_at_utc": "",
            }
            primary.append(base)
            if row.sku_id in second_set:
                # A fresh dict is load-bearing: primary decisions can never leak
                # into the independently distributed secondary packet.
                secondary.append(dict(base))

    expected_primary = expected_products * ATTRIBUTES_PER_PRODUCT
    expected_secondary = audit_products * ATTRIBUTES_PER_PRODUCT
    if len(primary) != expected_primary or len(secondary) != expected_secondary:
        raise RuntimeError("review packet cell counts drifted")
    manifest = {
        "version": VERSION,
        "status": "review_packets_built_before_human_decisions",
        "counts": {
            "products": expected_products,
            "attributes_per_product": ATTRIBUTES_PER_PRODUCT,
            "primary_cells": len(primary),
            "second_review_products": audit_products,
            "second_review_cells": len(secondary),
        },
        "second_review": {
            "seed": SECOND_REVIEW_SEED,
            "selector": "SHA-256 of '20260813-review\\0<sku_id>', ascending",
            "ordered_sku_ids": second_skus,
        },
        "invariants": {
            "all_primary_cells_included": True,
            "secondary_packet_generated_without_primary_decisions": True,
            "sft_or_grpo_predictions_included": False,
        },
    }
    return primary, secondary, manifest


def _write_csv(path: Path, values: Sequence[Mapping[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PACKET_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)
        handle.flush()
        os.fsync(handle.fileno())


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


def publish_review_packets(
    *,
    output_dir: str | Path,
    primary: Sequence[Mapping[str, str]],
    secondary: Sequence[Mapping[str, str]],
    plan_manifest: Mapping[str, Any],
    frontier_identity: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"review packet output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"review packet parent does not exist: {output_dir.parent}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    ).resolve()
    try:
        primary_path = staging / "primary-review.csv"
        secondary_path = staging / "secondary-review.csv"
        manifest_path = staging / "manifest.json"
        _write_csv(primary_path, primary)
        _write_csv(secondary_path, secondary)
        manifest = {
            **dict(plan_manifest),
            "version": PACKET_BUNDLE_VERSION,
            "status": "primary_and_blinded_secondary_packets_published",
            "frontier_identity": dict(frontier_identity),
            "files": {
                "primary_review": _identity(primary_path),
                "secondary_review": _identity(secondary_path),
            },
            "publication": {"exclusive": True, "atomic": True},
        }
        _write_json(manifest_path, manifest)
        if {path.name for path in staging.iterdir()} != PACKET_FILES:
            raise RuntimeError("review packet staging bundle has unexpected files")
        os.rename(staging, output_dir)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _timestamp(value: str, *, cell_id: str) -> str:
    if not value:
        raise ValueError(f"{cell_id} has no reviewed_at_utc")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{cell_id} has invalid reviewed_at_utc") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{cell_id} reviewed_at_utc must be timezone-aware")
    return value


def _proposed_label(row: Mapping[str, str]) -> AttributeLabel:
    return AttributeLabel(
        status=LabelStatus(row["frontier_status"]),
        value=json.loads(row["frontier_value_json"]),
    )


def _reviewed_label(row: Mapping[str, str], pack: Pack) -> AttributeLabel:
    proposed = _proposed_label(row)
    decision = row.get("decision", "").strip()
    if decision == "accept":
        if any(row.get(name, "").strip() for name in ("corrected_status", "corrected_value_json")):
            raise ValueError(f"{row['cell_id']} accept decision carries a correction")
        return proposed
    if decision != "correct":
        raise ValueError(f"{row['cell_id']} decision must be accept or correct")
    status_text = row.get("corrected_status", "").strip()
    if status_text not in {status.value for status in LabelStatus}:
        raise ValueError(f"{row['cell_id']} has invalid corrected_status")
    raw_value = row.get("corrected_value_json", "").strip()
    if not raw_value:
        raise ValueError(f"{row['cell_id']} correction has no corrected_value_json")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row['cell_id']} corrected_value_json is invalid") from exc
    corrected = AttributeLabel(status=LabelStatus(status_text), value=value)
    if corrected.key() == proposed.key():
        raise ValueError(f"{row['cell_id']} correction does not change the label")
    if not row.get("rationale", "").strip():
        raise ValueError(f"{row['cell_id']} correction has no rationale")
    attribute = row["attribute"]
    record_value = (
        corrected.value
        if corrected.status is LabelStatus.LABELED
        else None
        if corrected.status is LabelStatus.NOT_APPLICABLE
        else [pack.unknown_token]
        if pack.specs[attribute].kind == "multi"
        else pack.unknown_token
    )
    checked = verify_record({attribute: record_value}, pack)
    # Per-cell partial records are expected to fail whole-record schema checks;
    # vocabulary errors for the corrected field are not.
    if any(error.startswith("vocab") and f"{attribute}:" in error for error in checked.errors):
        raise ValueError(f"{row['cell_id']} correction is outside controlled vocabulary")
    return corrected


def import_completed_review(
    *,
    expected_packet: Sequence[Mapping[str, str]],
    completed_packet: Sequence[Mapping[str, str]],
    pack: Pack,
    review_role: str,
) -> dict[str, Any]:
    """Validate an explicit human decision for every expected packet cell."""

    expected = {row["cell_id"]: row for row in expected_packet}
    if len(expected) != len(expected_packet):
        raise ValueError("expected review packet contains duplicate cell IDs")
    completed: dict[str, Mapping[str, str]] = {}
    for row in completed_packet:
        cell_id = row.get("cell_id")
        if cell_id in completed:
            raise ValueError(f"duplicate completed review cell: {cell_id}")
        completed[str(cell_id)] = row
    if set(completed) != set(expected):
        missing = sorted(set(expected) - set(completed))
        extra = sorted(set(completed) - set(expected))
        raise ValueError(f"review cell membership drifted: missing={missing[:3]} extra={extra[:3]}")

    decisions = []
    correction_count = 0
    reviewer_ids: Counter[str] = Counter()
    for cell_id, expected_row in expected.items():
        row = completed[cell_id]
        for name in IMMUTABLE_COLUMNS:
            if str(row.get(name, "")) != str(expected_row.get(name, "")):
                raise ValueError(f"{cell_id} immutable review column drifted: {name}")
        reviewer = row.get("reviewer_id", "").strip()
        if not reviewer:
            raise ValueError(f"{cell_id} has no reviewer_id")
        reviewed_at = _timestamp(row.get("reviewed_at_utc", ""), cell_id=cell_id)
        label = _reviewed_label(row, pack)
        corrected = row["decision"].strip() == "correct"
        correction_count += corrected
        reviewer_ids[reviewer] += 1
        decisions.append(
            {
                "review_role": review_role,
                "cell_id": cell_id,
                "membership_order": int(row["membership_order"]),
                "sku_id": row["sku_id"],
                "attribute": row["attribute"],
                "reviewer_id": reviewer,
                "decision": row["decision"].strip(),
                "proposed_label": _proposed_label(row).model_dump(mode="json"),
                "final_label": label.model_dump(mode="json"),
                "rationale": row.get("rationale", "").strip() or None,
                "reviewed_at_utc": reviewed_at,
            }
        )
    return {
        "version": VERSION,
        "status": f"{review_role}_complete",
        "review_role": review_role,
        "counts": {
            "cells": len(decisions),
            "explicit_decisions": len(decisions),
            "corrections": correction_count,
            "reviewers": len(reviewer_ids),
        },
        "reviewer_cell_counts": dict(sorted(reviewer_ids.items())),
        "decisions": decisions,
    }


def compare_independent_reviews(
    *, primary_review: Mapping[str, Any], second_review: Mapping[str, Any]
) -> dict[str, Any]:
    primary = {item["cell_id"]: item for item in primary_review["decisions"]}
    second = {item["cell_id"]: item for item in second_review["decisions"]}
    if not set(second).issubset(primary):
        raise ValueError("second-review cells are not a subset of primary review")
    disagreements = []
    agreements = 0
    for cell_id, secondary in second.items():
        first = primary[cell_id]
        if first["reviewer_id"] == secondary["reviewer_id"]:
            raise ValueError(f"{cell_id} second reviewer is not independent")
        first_label = AttributeLabel.model_validate(first["final_label"])
        second_label = AttributeLabel.model_validate(secondary["final_label"])
        if first_label.key() == second_label.key():
            agreements += 1
        else:
            disagreements.append(
                {
                    "cell_id": cell_id,
                    "membership_order": first["membership_order"],
                    "sku_id": first["sku_id"],
                    "attribute": first["attribute"],
                    "primary_reviewer_id": first["reviewer_id"],
                    "second_reviewer_id": secondary["reviewer_id"],
                    "primary_label": first["final_label"],
                    "second_label": secondary["final_label"],
                }
            )
    total = len(second)
    return {
        "version": VERSION,
        "status": "independent_reviews_compared_before_adjudication",
        "counts": {
            "audited_cells": total,
            "agreements": agreements,
            "disagreements": len(disagreements),
        },
        "agreement_before_adjudication": agreements / total if total else 0.0,
        "disagreements": disagreements,
    }


def build_adjudication_packet(comparison: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "cell_id": item["cell_id"],
            "membership_order": str(item["membership_order"]),
            "sku_id": item["sku_id"],
            "attribute": item["attribute"],
            "primary_reviewer_id": item["primary_reviewer_id"],
            "second_reviewer_id": item["second_reviewer_id"],
            "primary_label_json": _canonical_json(item["primary_label"]),
            "second_label_json": _canonical_json(item["second_label"]),
            "adjudicator_id": "",
            "decision": "",
            "custom_status": "",
            "custom_value_json": "",
            "rationale": "",
            "adjudicated_at_utc": "",
        }
        for item in comparison["disagreements"]
    ]


def import_adjudication(
    *,
    comparison: Mapping[str, Any],
    completed_packet: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    expected = {item["cell_id"]: item for item in comparison["disagreements"]}
    completed = {str(item.get("cell_id")): item for item in completed_packet}
    if len(completed) != len(completed_packet):
        raise ValueError("adjudication packet contains duplicate cell IDs")
    if set(completed) != set(expected):
        raise ValueError("adjudication does not resolve every disagreement exactly once")
    decisions = []
    for cell_id, disagreement in expected.items():
        row = completed[cell_id]
        adjudicator = row.get("adjudicator_id", "").strip()
        if not adjudicator:
            raise ValueError(f"{cell_id} has no adjudicator_id")
        if adjudicator in {
            disagreement["primary_reviewer_id"],
            disagreement["second_reviewer_id"],
        }:
            raise ValueError(f"{cell_id} adjudicator is not independent")
        decision = row.get("decision", "").strip()
        if decision == "primary":
            final = disagreement["primary_label"]
        elif decision == "second":
            final = disagreement["second_label"]
        elif decision == "custom":
            status = row.get("custom_status", "").strip()
            value_raw = row.get("custom_value_json", "").strip()
            if status not in {item.value for item in LabelStatus} or not value_raw:
                raise ValueError(f"{cell_id} custom adjudication is incomplete")
            final = AttributeLabel(
                status=LabelStatus(status), value=json.loads(value_raw)
            ).model_dump(mode="json")
        else:
            raise ValueError(f"{cell_id} has invalid adjudication decision")
        if not row.get("rationale", "").strip():
            raise ValueError(f"{cell_id} adjudication has no rationale")
        timestamp = _timestamp(row.get("adjudicated_at_utc", ""), cell_id=cell_id)
        decisions.append(
            {
                "review_role": "adjudication",
                "cell_id": cell_id,
                "membership_order": disagreement["membership_order"],
                "sku_id": disagreement["sku_id"],
                "attribute": disagreement["attribute"],
                "adjudicator_id": adjudicator,
                "decision": decision,
                "primary_label": disagreement["primary_label"],
                "second_label": disagreement["second_label"],
                "final_label": final,
                "rationale": row["rationale"].strip(),
                "adjudicated_at_utc": timestamp,
            }
        )
    return {
        "version": VERSION,
        "status": "all_disagreements_adjudicated",
        "counts": {"resolved": len(decisions), "unresolved": 0},
        "decisions": decisions,
    }


def _support(rows: Sequence[Row], pack: Pack) -> dict[str, Any]:
    attributes = {}
    shortfalls = []
    for attribute in pack.field_names:
        status_counts = Counter(row.labels[attribute].status.value for row in rows)
        value_counts: Counter[str] = Counter()
        for row in rows:
            label = row.labels[attribute]
            if label.status is not LabelStatus.LABELED:
                continue
            values = label.value if isinstance(label.value, list) else [label.value]
            value_counts.update(str(value) for value in values)
        all_values = {value: value_counts[value] for value in pack.specs[attribute].values}
        all_statuses = {status.value: status_counts[status.value] for status in LabelStatus}
        attributes[attribute] = {"status_counts": all_statuses, "value_counts": all_values}
        for status, count in all_statuses.items():
            if count < SUPPORT_TARGET:
                shortfalls.append({"attribute": attribute, "kind": "status", "value": status, "count": count})
        for value, count in all_values.items():
            if count < SUPPORT_TARGET:
                shortfalls.append({"attribute": attribute, "kind": "value", "value": value, "count": count})
    return {
        "target_per_attribute_status_or_value": SUPPORT_TARGET,
        "attributes": attributes,
        "shortfalls": shortfalls,
        "shortfalls_change_membership": False,
    }


def finalize_reviewed_bundle(
    *,
    output_dir: str | Path,
    frontier_rows: Sequence[Row | Mapping[str, Any]],
    primary_review: Mapping[str, Any],
    second_review: Mapping[str, Any],
    comparison: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    pack: Pack,
    frontier_identity: Mapping[str, Any],
    expected_products: int = EXPECTED_PRODUCTS,
) -> dict[str, Any]:
    """Publish reviewed rows only after all human and verifier gates pass."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"reviewed output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"reviewed output parent does not exist: {output_dir.parent}")
    rows = _rows(frontier_rows, expected_products=expected_products)
    expected_primary_cells = expected_products * ATTRIBUTES_PER_PRODUCT
    if primary_review.get("counts", {}).get("explicit_decisions") != expected_primary_cells:
        raise ValueError("primary review is not complete for every cell")
    expected_secondary_cells = comparison.get("counts", {}).get("audited_cells")
    if expected_secondary_cells != SECOND_REVIEW_CELLS:
        raise ValueError(
            f"independent review has {expected_secondary_cells} cells; "
            f"expected {SECOND_REVIEW_CELLS}"
        )
    if second_review.get("counts", {}).get("explicit_decisions") != expected_secondary_cells:
        raise ValueError("second review is incomplete")
    if adjudication.get("counts", {}).get("unresolved") != 0:
        raise ValueError("adjudication still has unresolved cells")
    if adjudication.get("counts", {}).get("resolved") != comparison.get("counts", {}).get("disagreements"):
        raise ValueError("adjudication count does not match review disagreements")

    primary_by_cell = {item["cell_id"]: item for item in primary_review["decisions"]}
    adjudicated = {item["cell_id"]: item for item in adjudication["decisions"]}
    final_rows: list[Row] = []
    final_change_events = []
    for membership_order, frontier in enumerate(rows, start=1):
        reviewed = frontier.model_copy(deep=True)
        corrected_timestamps = []
        for attribute in pack.field_names:
            cell_id = f"{membership_order:04d}::{attribute}"
            primary = primary_by_cell[cell_id]
            final_data = adjudicated.get(cell_id, primary)["final_label"]
            final_label = AttributeLabel.model_validate(final_data)
            if final_label.key() != frontier.labels[attribute].key():
                event = adjudicated.get(cell_id, primary)
                timestamp = event.get("adjudicated_at_utc") or event.get("reviewed_at_utc")
                corrected_timestamps.append(timestamp)
                final_change_events.append(
                    {
                        "review_role": "final_change_from_frontier",
                        "cell_id": cell_id,
                        "sku_id": frontier.sku_id,
                        "attribute": attribute,
                        "frontier_label": frontier.labels[attribute].model_dump(mode="json"),
                        "final_label": final_label.model_dump(mode="json"),
                        "source_decision": event,
                    }
                )
            reviewed.labels[attribute] = final_label
        verified = verify_record(reviewed.to_verifier_record(pack), pack)
        if not verified.schema_valid or not verified.vocab_valid or verified.rule_violations:
            raise ValueError(
                f"final reviewed SKU {reviewed.sku_id} failed verifier: "
                f"errors={verified.errors[:3]} rules={verified.rule_violations[:3]}"
            )
        if corrected_timestamps:
            reviewed.provenance.human_corrected = True
            reviewed.provenance.corrected_at = max(corrected_timestamps)
        final_rows.append(reviewed)

    support = _support(final_rows, pack)
    decisions = (
        list(primary_review["decisions"])
        + list(second_review["decisions"])
        + list(adjudication["decisions"])
        + final_change_events
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    ).resolve()
    try:
        reviewed_path = staging / "reviewed.jsonl"
        decisions_path = staging / "decisions.jsonl"
        support_path = staging / "support.json"
        manifest_path = staging / "manifest.json"
        _write_jsonl(reviewed_path, [row.model_dump(mode="json") for row in final_rows])
        _write_jsonl(decisions_path, decisions)
        _write_json(support_path, support)
        manifest = {
            "version": FINAL_BUNDLE_VERSION,
            "status": "human_review_complete_ready_for_final_freeze",
            "frontier_identity": dict(frontier_identity),
            "counts": {
                "products": len(final_rows),
                "primary_reviewed_cells": primary_review["counts"]["explicit_decisions"],
                "second_reviewed_products": expected_secondary_cells // ATTRIBUTES_PER_PRODUCT,
                "second_reviewed_cells": expected_secondary_cells,
                "pre_adjudication_agreements": comparison["counts"]["agreements"],
                "pre_adjudication_disagreements": comparison["counts"]["disagreements"],
                "adjudicated_cells": adjudication["counts"]["resolved"],
                "unresolved_cells": 0,
                "final_changes_from_frontier": len(final_change_events),
            },
            "agreement_before_adjudication": comparison["agreement_before_adjudication"],
            "reviewers": {
                "primary": primary_review["reviewer_cell_counts"],
                "second": second_review["reviewer_cell_counts"],
                "adjudicators": sorted(
                    {item["adjudicator_id"] for item in adjudication["decisions"]}
                ),
            },
            "files": {
                "reviewed": _identity(reviewed_path),
                "decisions": _identity(decisions_path),
                "support": _identity(support_path),
            },
            "invariants": {
                "all_6000_primary_cells_reviewed": expected_primary_cells == PRIMARY_CELLS,
                "all_600_second_review_cells_reviewed": expected_secondary_cells == SECOND_REVIEW_CELLS,
                "second_review_independent": True,
                "all_disagreements_adjudicated": True,
                "all_final_rows_schema_vocab_rule_valid": True,
                "support_shortfalls_disclosed_without_membership_changes": True,
                "sft_or_grpo_predictions_used": False,
                "published_exclusively_and_atomically": True,
            },
        }
        _write_json(manifest_path, manifest)
        if {path.name for path in staging.iterdir()} != FINAL_FILES:
            raise RuntimeError("reviewed staging bundle has unexpected files")
        os.rename(staging, output_dir)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
