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
from training.run2_confirmation_freeze import freeze_confirmation_bundle
from training.split_sft import group_key
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def _rows(pack):
    labels = {
        name: AttributeLabel(status=LabelStatus.UNKNOWN) for name in pack.field_names
    }
    agreement = {name: 1.0 for name in pack.field_names}
    return [
        Row(
            sku_id=f"shopify:approved-{index % 8:02d}.example:{index}",
            source=f"shopify:approved-{index % 8:02d}.example",
            split="eval",
            input=RowInput(
                title=f"Confirmation Dress {index}",
                description="Human-reviewed listing.",
                brand=f"Confirmation Brand {index}",
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
        for index in range(400)
    ]


def _selection(rows):
    return {
        "status": "confirmation_membership_selected_before_labeling",
        "selected": [
            {"selection_order": index, "sku_id": row.sku_id}
            for index, row in enumerate(rows, start=1)
        ],
        "invariants": {"membership_uses_labels_or_model_outputs": False},
    }


def _review():
    return {
        "status": "human_review_complete_ready_for_final_freeze",
        "counts": {
            "products": 400,
            "primary_reviewed_cells": 6000,
            "second_reviewed_products": 40,
            "second_reviewed_cells": 600,
            "pre_adjudication_agreements": 600,
            "pre_adjudication_disagreements": 0,
            "adjudicated_cells": 0,
            "unresolved_cells": 0,
            "final_changes_from_frontier": 0,
        },
        "agreement_before_adjudication": 1.0,
        "invariants": {
            "all_6000_primary_cells_reviewed": True,
            "all_600_second_review_cells_reviewed": True,
            "second_review_independent": True,
            "all_disagreements_adjudicated": True,
            "all_final_rows_schema_vocab_rule_valid": True,
            "support_shortfalls_disclosed_without_membership_changes": True,
        },
    }


def _support(pack):
    return {
        "target_per_attribute_status_or_value": 8,
        "attributes": {name: {} for name in pack.field_names},
        "shortfalls": [],
        "shortfalls_change_membership": False,
    }


def _identity(name):
    return {"path": name, "bytes": 1, "sha256": "a" * 64}


def _lineage():
    return {
        "source_terms_audit": _identity("terms.json"),
        "acquisition_manifest": _identity("acquisition.json"),
        "selection_manifest": _identity("selection.json"),
        "frontier_labeling_manifest": _identity("frontier.json"),
        "review_manifest": _identity("review.json"),
        "reviewed_dataset": _identity("reviewed.jsonl"),
    }


def _prior(rows):
    prior_skus = frozenset(f"prior-{index}" for index in range(4000))
    prior_families = frozenset(f"prior-family-{index}" for index in range(4000))
    assert not prior_skus.intersection(row.sku_id for row in rows)
    assert not prior_families.intersection(group_key(row) for row in rows)
    return prior_skus, prior_families


def _parent():
    return json.loads((ROOT / "runs/grpo-run2-data-role-manifest.json").read_text())


def _freeze(tmp_path, pack, **overrides):
    rows = overrides.pop("rows", _rows(pack))
    prior_skus, prior_families = _prior(rows)
    data_parent = tmp_path / "data"
    run_parent = tmp_path / "runs"
    data_parent.mkdir(parents=True)
    run_parent.mkdir(parents=True)
    kwargs = {
        "output_dir": data_parent / "confirmation_run2_v1",
        "role_output": run_parent / "grpo-run2-data-role-manifest-confirmation-assigned.json",
        "reviewed_rows": rows,
        "selection_manifest": _selection(rows),
        "review_manifest": _review(),
        "support_report": _support(pack),
        "source_gate_result": {"passed": True, "approved_store_count": 8},
        "prior_sku_ids": prior_skus,
        "prior_family_keys": prior_families,
        "pack": pack,
        "lineage": _lineage(),
        "parent_role_manifest": _parent(),
        "parent_role_identity": _identity("grpo-run2-data-role-manifest.json"),
        "code_identity": {"git_commit": "b" * 40},
        "frozen_at_utc": "2026-08-13T15:00:00Z",
    }
    kwargs.update(overrides)
    return freeze_confirmation_bundle(**kwargs), kwargs


def test_successful_freeze_preserves_exact_order_and_assigns_role(pack, tmp_path) -> None:
    (manifest, role), kwargs = _freeze(tmp_path, pack)
    output = Path(kwargs["output_dir"])
    role_path = Path(kwargs["role_output"])

    assert manifest["status"] == "confirmation_frozen_sealed_before_final_recipe_lock"
    assert manifest["summary"]["products"] == 400
    assert manifest["summary"]["prior_exact_sku_overlap"] == 0
    assert manifest["summary"]["prior_family_overlap"] == 0
    assert manifest["exact_row_order"] == [row.sku_id for row in kwargs["reviewed_rows"]]
    assert {path.name for path in output.iterdir()} == {"eval.jsonl", "manifest.json"}
    assert role_path.exists()
    assert role["version"] == "grpo-run2-data-role-manifest-v2"
    assert role["status"] == "development_and_confirmation_roles_locked"
    assert role["final_confirmation"]["assigned"] is True
    assert role["final_confirmation"]["model_outputs_generated"] is False
    assert role["phase_e_gate"]["passed"] is True


def test_reviewed_membership_or_order_drift_blocks_without_outputs(pack, tmp_path) -> None:
    rows = _rows(pack)
    selection = _selection(rows)
    selection["selected"][0], selection["selected"][1] = (
        selection["selected"][1],
        selection["selected"][0],
    )

    with pytest.raises(ValueError, match="order or membership"):
        _freeze(tmp_path, pack, rows=rows, selection_manifest=selection)
    assert not (tmp_path / "data" / "confirmation_run2_v1").exists()


def test_prior_exact_sku_or_family_overlap_blocks(pack, tmp_path) -> None:
    rows = _rows(pack)
    prior_skus, prior_families = _prior(rows)
    prior_skus = frozenset({*prior_skus, rows[0].sku_id})
    # Keep the locked universe count exactly 4,000 while introducing overlap.
    prior_skus = frozenset(sorted(prior_skus)[1:])
    if rows[0].sku_id not in prior_skus:
        prior_skus = frozenset({*list(prior_skus)[1:], rows[0].sku_id})

    with pytest.raises(ValueError, match="exact-SKU overlap"):
        _freeze(
            tmp_path,
            pack,
            rows=rows,
            prior_sku_ids=prior_skus,
            prior_family_keys=prior_families,
        )


def test_incomplete_review_or_failed_permission_gate_blocks(pack, tmp_path) -> None:
    review = _review()
    review["counts"]["primary_reviewed_cells"] = 5999
    with pytest.raises(ValueError, match="primary_reviewed_cells"):
        _freeze(tmp_path, pack, review_manifest=review)

    other = tmp_path / "other"
    with pytest.raises(ValueError, match="permission gate"):
        _freeze(
            other,
            pack,
            source_gate_result={"passed": False, "approved_store_count": 0},
        )


def test_invalid_final_vocabulary_blocks(pack, tmp_path) -> None:
    rows = _rows(pack)
    attribute = pack.field_names[0]
    rows[0].labels[attribute] = AttributeLabel(
        status=LabelStatus.LABELED, value="not-in-vocabulary"
    )

    with pytest.raises(ValueError, match="failed final verifier"):
        _freeze(tmp_path, pack, rows=rows)


def test_output_collision_fails_before_publication(pack, tmp_path) -> None:
    output = tmp_path / "data" / "confirmation_run2_v1"
    output.mkdir(parents=True)
    role_parent = tmp_path / "runs"
    role_parent.mkdir()
    rows = _rows(pack)
    prior_skus, prior_families = _prior(rows)

    with pytest.raises(FileExistsError, match="already exists"):
        freeze_confirmation_bundle(
            output_dir=output,
            role_output=role_parent / "grpo-run2-data-role-manifest-confirmation-assigned.json",
            reviewed_rows=rows,
            selection_manifest=_selection(rows),
            review_manifest=_review(),
            support_report=_support(pack),
            source_gate_result={"passed": True, "approved_store_count": 8},
            prior_sku_ids=prior_skus,
            prior_family_keys=prior_families,
            pack=pack,
            lineage=_lineage(),
            parent_role_manifest=_parent(),
            parent_role_identity=_identity("parent.json"),
            code_identity={"git_commit": "b" * 40},
            frozen_at_utc="2026-08-13T15:00:00Z",
        )


def test_freeze_contains_no_confirmation_metric(pack, tmp_path) -> None:
    (manifest, role), _kwargs = _freeze(tmp_path, pack)
    encoded = json.dumps({"manifest": manifest, "role": role})

    assert "macro_f1" not in encoded
    assert "confirmation_metrics_calculated\": true" not in encoded.lower()
    assert manifest["decision_boundary"]["aggregate_confirmation_metrics_calculated"] is False


def test_parent_role_must_be_unassigned_history(pack, tmp_path) -> None:
    parent = _parent()
    parent["final_confirmation"]["assigned"] = True

    with pytest.raises(ValueError, match="already assigns"):
        _freeze(tmp_path, pack, parent_role_manifest=parent)
