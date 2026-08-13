from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from labeling.records import (
    AttributeLabel,
    LabelStatus,
    Provenance,
    Row,
    RowInput,
    SelfConsistency,
)
from training.run2_confirmation_review import (
    build_adjudication_packet,
    build_review_packets,
    compare_independent_reviews,
    finalize_reviewed_bundle,
    import_adjudication,
    import_completed_review,
    publish_review_packets,
    select_second_review_skus,
)
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def _unknown_label():
    return AttributeLabel(status=LabelStatus.UNKNOWN)


def _frontier(pack, n=400):
    agreement = {name: 1.0 for name in pack.field_names}
    labels = {name: _unknown_label() for name in pack.field_names}
    return [
        Row(
            sku_id=f"shopify:store-{index % 10:02d}.example:{index}",
            source=f"shopify:store-{index % 10:02d}.example",
            split="eval",
            input=RowInput(
                title=f"Dress {index}",
                description="Description for human review.",
                raw_tags=["dress"],
                brand=f"Brand {index}",
                category="Dress",
            ),
            labels=deepcopy(labels),
            provenance=Provenance(
                labeler="gpt-5.6-luna@prelabel-v1",
                prompt_version="prelabel-v1",
                self_consistency=SelfConsistency(k=5, agreement=agreement),
                frontier_labels=deepcopy(labels),
            ),
        )
        for index in range(n)
    ]


def _attempts(frontier, pack):
    labels = {
        name: _unknown_label().model_dump(mode="json") for name in pack.field_names
    }
    return [
        {
            "sku_id": row.sku_id,
            "variant": variant,
            "usable": True,
            "labels": deepcopy(labels),
        }
        for row in frontier
        for variant in range(5)
    ]


def _fill(packet, reviewer):
    completed = deepcopy(packet)
    for row in completed:
        row["reviewer_id"] = reviewer
        row["decision"] = "accept"
        row["reviewed_at_utc"] = "2026-08-13T12:00:00Z"
    return completed


def test_packets_cover_all_cells_and_deterministic_blinded_sample(pack) -> None:
    frontier = _frontier(pack)
    primary, secondary, manifest = build_review_packets(
        frontier_rows=frontier, attempts=_attempts(frontier, pack), pack=pack
    )

    assert len(primary) == 6000
    assert len(secondary) == 600
    assert len(manifest["second_review"]["ordered_sku_ids"]) == 40
    assert {row["sku_id"] for row in secondary} == set(
        manifest["second_review"]["ordered_sku_ids"]
    )
    assert all(not row["decision"] and not row["reviewer_id"] for row in secondary)
    assert all("prediction" not in key for row in primary for key in row)
    assert select_second_review_skus(frontier) == manifest["second_review"]["ordered_sku_ids"]


def test_packet_publication_is_atomic_and_collision_safe(pack, tmp_path) -> None:
    frontier = _frontier(pack)
    primary, secondary, plan = build_review_packets(
        frontier_rows=frontier, attempts=_attempts(frontier, pack), pack=pack
    )
    output = tmp_path / "review-packets"

    manifest = publish_review_packets(
        output_dir=output,
        primary=primary,
        secondary=secondary,
        plan_manifest=plan,
        frontier_identity={"sha256": "a" * 64, "rows": 400},
    )

    assert manifest["publication"] == {"exclusive": True, "atomic": True}
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "primary-review.csv",
        "secondary-review.csv",
    }
    with pytest.raises(FileExistsError, match="already exists"):
        publish_review_packets(
            output_dir=output,
            primary=primary,
            secondary=secondary,
            plan_manifest=plan,
            frontier_identity={"sha256": "a" * 64},
        )


def test_primary_and_secondary_require_every_explicit_cell(pack) -> None:
    frontier = _frontier(pack)
    primary, secondary, _plan = build_review_packets(
        frontier_rows=frontier, attempts=_attempts(frontier, pack), pack=pack
    )

    first = import_completed_review(
        expected_packet=primary,
        completed_packet=_fill(primary, "reviewer-primary"),
        pack=pack,
        review_role="primary_review",
    )
    second = import_completed_review(
        expected_packet=secondary,
        completed_packet=_fill(secondary, "reviewer-secondary"),
        pack=pack,
        review_role="second_review",
    )

    assert first["counts"]["explicit_decisions"] == 6000
    assert second["counts"]["explicit_decisions"] == 600
    with pytest.raises(ValueError, match="membership drifted"):
        import_completed_review(
            expected_packet=primary,
            completed_packet=_fill(primary, "reviewer")[:-1],
            pack=pack,
            review_role="primary_review",
        )


def test_correction_requires_change_rationale_timestamp_and_vocab(pack) -> None:
    frontier = _frontier(pack)
    primary, _secondary, _plan = build_review_packets(
        frontier_rows=frontier, attempts=_attempts(frontier, pack), pack=pack
    )
    completed = _fill(primary, "reviewer-primary")
    row = completed[0]
    row["decision"] = "correct"
    row["corrected_status"] = "labeled"
    row["corrected_value_json"] = '"not-in-vocabulary"'
    row["rationale"] = "Listing states this value."

    with pytest.raises(ValueError, match="outside controlled vocabulary"):
        import_completed_review(
            expected_packet=primary,
            completed_packet=completed,
            pack=pack,
            review_role="primary_review",
        )

    row["corrected_status"] = "unknown"
    row["corrected_value_json"] = "null"
    with pytest.raises(ValueError, match="does not change"):
        import_completed_review(
            expected_packet=primary,
            completed_packet=completed,
            pack=pack,
            review_role="primary_review",
        )


def test_second_reviewer_must_be_independent(pack) -> None:
    frontier = _frontier(pack)
    primary, secondary, _plan = build_review_packets(
        frontier_rows=frontier, attempts=_attempts(frontier, pack), pack=pack
    )
    first = import_completed_review(
        expected_packet=primary,
        completed_packet=_fill(primary, "same-reviewer"),
        pack=pack,
        review_role="primary_review",
    )
    second = import_completed_review(
        expected_packet=secondary,
        completed_packet=_fill(secondary, "same-reviewer"),
        pack=pack,
        review_role="second_review",
    )

    with pytest.raises(ValueError, match="not independent"):
        compare_independent_reviews(primary_review=first, second_review=second)


def test_disagreement_requires_independent_adjudication(pack) -> None:
    frontier = _frontier(pack)
    primary, secondary, _plan = build_review_packets(
        frontier_rows=frontier, attempts=_attempts(frontier, pack), pack=pack
    )
    first_completed = _fill(primary, "reviewer-primary")
    second_completed = _fill(secondary, "reviewer-secondary")
    disputed = second_completed[0]
    attribute = disputed["attribute"]
    valid_value = pack.specs[attribute].values[0]
    disputed["decision"] = "correct"
    disputed["corrected_status"] = "labeled"
    disputed["corrected_value_json"] = (
        json.dumps([valid_value])
        if pack.specs[attribute].kind == "multi"
        else json.dumps(valid_value)
    )
    disputed["rationale"] = "Independent reviewer found explicit listing evidence."
    first = import_completed_review(
        expected_packet=primary,
        completed_packet=first_completed,
        pack=pack,
        review_role="primary_review",
    )
    second = import_completed_review(
        expected_packet=secondary,
        completed_packet=second_completed,
        pack=pack,
        review_role="second_review",
    )
    comparison = compare_independent_reviews(primary_review=first, second_review=second)

    assert comparison["counts"] == {
        "audited_cells": 600,
        "agreements": 599,
        "disagreements": 1,
    }
    packet = build_adjudication_packet(comparison)
    assert len(packet) == 1
    packet[0]["adjudicator_id"] = "reviewer-primary"
    packet[0]["decision"] = "primary"
    packet[0]["rationale"] = "Resolved from source evidence."
    packet[0]["adjudicated_at_utc"] = "2026-08-13T13:00:00Z"
    with pytest.raises(ValueError, match="not independent"):
        import_adjudication(comparison=comparison, completed_packet=packet)

    packet[0]["adjudicator_id"] = "adjudicator-independent"
    adjudication = import_adjudication(comparison=comparison, completed_packet=packet)
    assert adjudication["counts"] == {"resolved": 1, "unresolved": 0}


def test_all_accept_reviews_finalize_and_disclose_support_shortfalls(
    pack, tmp_path
) -> None:
    frontier = _frontier(pack)
    primary, secondary, _plan = build_review_packets(
        frontier_rows=frontier, attempts=_attempts(frontier, pack), pack=pack
    )
    first = import_completed_review(
        expected_packet=primary,
        completed_packet=_fill(primary, "reviewer-primary"),
        pack=pack,
        review_role="primary_review",
    )
    second = import_completed_review(
        expected_packet=secondary,
        completed_packet=_fill(secondary, "reviewer-secondary"),
        pack=pack,
        review_role="second_review",
    )
    comparison = compare_independent_reviews(primary_review=first, second_review=second)
    adjudication = import_adjudication(comparison=comparison, completed_packet=[])
    output = tmp_path / "reviewed"

    manifest = finalize_reviewed_bundle(
        output_dir=output,
        frontier_rows=frontier,
        primary_review=first,
        second_review=second,
        comparison=comparison,
        adjudication=adjudication,
        pack=pack,
        frontier_identity={"sha256": "a" * 64, "rows": 400},
    )

    assert manifest["status"] == "human_review_complete_ready_for_final_freeze"
    assert manifest["counts"]["primary_reviewed_cells"] == 6000
    assert manifest["counts"]["second_reviewed_cells"] == 600
    assert manifest["counts"]["unresolved_cells"] == 0
    assert manifest["agreement_before_adjudication"] == 1.0
    assert (output / "support.json").exists()
    assert {path.name for path in output.iterdir()} == {
        "decisions.jsonl",
        "manifest.json",
        "reviewed.jsonl",
        "support.json",
    }


def test_finalization_blocks_incomplete_human_work(pack, tmp_path) -> None:
    frontier = _frontier(pack)
    primary, secondary, _plan = build_review_packets(
        frontier_rows=frontier, attempts=_attempts(frontier, pack), pack=pack
    )
    first = import_completed_review(
        expected_packet=primary,
        completed_packet=_fill(primary, "reviewer-primary"),
        pack=pack,
        review_role="primary_review",
    )
    second = import_completed_review(
        expected_packet=secondary,
        completed_packet=_fill(secondary, "reviewer-secondary"),
        pack=pack,
        review_role="second_review",
    )
    comparison = compare_independent_reviews(primary_review=first, second_review=second)
    adjudication = import_adjudication(comparison=comparison, completed_packet=[])
    first["counts"]["explicit_decisions"] = 5999

    with pytest.raises(ValueError, match="primary review is not complete"):
        finalize_reviewed_bundle(
            output_dir=tmp_path / "blocked",
            frontier_rows=frontier,
            primary_review=first,
            second_review=second,
            comparison=comparison,
            adjudication=adjudication,
            pack=pack,
            frontier_identity={"sha256": "a" * 64},
        )
    assert list(tmp_path.iterdir()) == []
