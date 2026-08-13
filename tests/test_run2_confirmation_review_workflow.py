from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from labeling.records import AttributeLabel, LabelStatus, Provenance, Row, RowInput, SelfConsistency
from training.run2_confirmation_review_workflow import (
    compare_review_workflow,
    import_adjudication_workflow,
    import_review_workflow,
    prepare_review_workflow,
)
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def _frontier_bundle(tmp_path: Path, pack):
    output = tmp_path / "frontier"
    output.mkdir()
    labels = {name: AttributeLabel(status=LabelStatus.UNKNOWN) for name in pack.field_names}
    agreement = {name: 1.0 for name in pack.field_names}
    row = Row(
        sku_id="shopify:store.example:1",
        source="shopify:store.example",
        split="eval",
        input=RowInput(title="Dress", description="Cotton dress", raw_tags=["dress"]),
        labels=deepcopy(labels),
        provenance=Provenance(
            labeler="gpt-5.6-luna@prelabel-v1",
            prompt_version="prelabel-v1",
            self_consistency=SelfConsistency(k=5, agreement=agreement),
            frontier_labels=deepcopy(labels),
        ),
    )
    frontier = output / "frontier.jsonl"
    frontier.write_text(json.dumps(row.model_dump(mode="json")) + "\n", encoding="utf-8")
    attempts = output / "attempts.jsonl"
    sample = {name: label.model_dump(mode="json") for name, label in labels.items()}
    attempts.write_text(
        "".join(
            json.dumps({"sku_id": row.sku_id, "variant": variant, "usable": True, "labels": sample}) + "\n"
            for variant in range(5)
        ),
        encoding="utf-8",
    )
    import hashlib

    def ident(path):
        raw = path.read_bytes()
        return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "frontier_labels_complete_awaiting_human_review",
                "files": {"frontier": ident(frontier), "attempts": ident(attempts)},
            }
        ),
        encoding="utf-8",
    )
    return output


def _fill_csv(source: Path, target: Path, reviewer: str):
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        row["reviewer_id"] = reviewer
        row["decision"] = "accept"
        row["reviewed_at_utc"] = "2026-08-13T00:00:00Z"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_and_import_preserve_packet_identities(tmp_path, pack):
    frontier = _frontier_bundle(tmp_path, pack)
    packets = tmp_path / "packets"
    manifest = prepare_review_workflow(
        frontier_dir=frontier,
        output_dir=packets,
        pack=pack,
        expected_products=1,
        audit_products=1,
    )
    assert manifest["counts"]["primary_cells"] == 15
    assert manifest["counts"]["second_review_cells"] == 15
    completed = tmp_path / "primary-completed.csv"
    _fill_csv(packets / "primary-review.csv", completed, "primary-reviewer")
    output = tmp_path / "primary.json"
    imported = import_review_workflow(
        expected_packet_path=packets / "primary-review.csv",
        completed_packet_path=completed,
        output_path=output,
        pack=pack,
        review_role="primary_review",
    )
    assert imported["counts"]["explicit_decisions"] == 15
    assert imported["workflow"]["completed_packet"]["sha256"]


def test_prepare_rejects_frontier_identity_drift(tmp_path, pack):
    frontier = _frontier_bundle(tmp_path, pack)
    (frontier / "attempts.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="attempts identity drifted"):
        prepare_review_workflow(
            frontier_dir=frontier,
            output_dir=tmp_path / "packets",
            pack=pack,
            expected_products=1,
            audit_products=1,
        )


def test_compare_publishes_empty_adjudication_packet_atomically(tmp_path):
    decision = {
        "cell_id": "0001::garment_category",
        "membership_order": 1,
        "sku_id": "shopify:store.example:1",
        "attribute": "garment_category",
        "reviewer_id": "primary",
        "final_label": {"status": "unknown", "value": None},
    }
    primary = {
        "decisions": [decision],
    }
    second_decision = {**decision, "reviewer_id": "secondary"}
    second = {"decisions": [second_decision]}
    primary_path = tmp_path / "primary.json"
    second_path = tmp_path / "second.json"
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    output = tmp_path / "comparison"
    manifest = compare_review_workflow(
        primary_review_path=primary_path,
        second_review_path=second_path,
        output_dir=output,
    )
    assert manifest["counts"]["disagreements"] == 0
    assert set(path.name for path in output.iterdir()) == {
        "comparison.json",
        "adjudication.csv",
        "manifest.json",
    }
    with pytest.raises(FileExistsError):
        compare_review_workflow(
            primary_review_path=primary_path,
            second_review_path=second_path,
            output_dir=output,
        )


def test_empty_adjudication_import_is_valid_when_reviews_agree(tmp_path):
    comparison = {
        "disagreements": [],
        "counts": {"audited_cells": 1, "agreements": 1, "disagreements": 0},
    }
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    packet = tmp_path / "adjudication.csv"
    packet.write_text(
        "cell_id,membership_order,sku_id,attribute,primary_reviewer_id,second_reviewer_id,"
        "primary_label_json,second_label_json,adjudicator_id,decision,custom_status,"
        "custom_value_json,rationale,adjudicated_at_utc\n",
        encoding="utf-8",
    )
    output = tmp_path / "adjudication.json"
    result = import_adjudication_workflow(
        comparison_path=comparison_path,
        completed_packet_path=packet,
        output_path=output,
    )
    assert result["counts"] == {"resolved": 0, "unresolved": 0}
