from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from labeling.records import Provenance, Row, RowInput, write_jsonl
from training.audit_data_boundaries import (
    build_boundary_audit,
    load_audit_inputs,
    sha256_file,
    write_exclusive_atomic_json,
)


def _row(sku: str, *, brand: str, title: str, split: str = "train") -> Row:
    return Row(
        sku_id=sku,
        source="synthetic",
        split=split,
        input=RowInput(title=title, brand=brand),
        provenance=Provenance(
            labeler="test@v1",
            prompt_version="test-v1",
        ),
    )


def _inputs() -> dict[str, dict]:
    return {
        name: {"path": f"{name}.json", "bytes": 1, "sha256": name * 4}
        for name in (
            "sft_manifest",
            "sft_source",
            "grpo_pool_manifest",
            "grpo_pool_dataset",
            "grpo_run1_rollouts",
            "probe_dataset",
            "legacy_frozen_dataset",
        )
    }


def test_boundary_audit_reports_sku_and_family_leakage_deterministically():
    rows = {
        "train-a": _row("train-a", brand="A", title="Classic Tee - Blue"),
        "train-b": _row("train-b", brand="B", title="Trouser"),
        "val-a": _row("val-a", brand="V", title="Validation Dress"),
        "probe-a": _row("probe-a", brand="P", title="Probe Shoe", split="probe"),
        "eval-a": _row("eval-a", brand="E", title="Eval Coat", split="eval"),
    }
    collections = {
        "sft_train": ["train-a", "train-b"],
        "sft_validation": ["val-a"],
        "grpo_pool_cap4": ["train-a", "val-a"],
        "grpo_run1_trained": ["val-a"],
        "probe_100": ["probe-a"],
        "legacy_frozen_300": ["eval-a"],
    }
    kwargs = {
        "collections": collections,
        "family_rows": rows,
        "inputs": _inputs(),
        "source_details": {"fixture": True},
        "code": {"git_commit": "a" * 40},
    }

    first = build_boundary_audit(**kwargs)
    second = build_boundary_audit(**kwargs)

    assert first == second
    assert first["status"] == "issues_found"
    assert first["headline_findings"]["grpo_pool_validation_sku_count"] == 1
    assert first["headline_findings"]["grpo_pool_validation_skus"] == ["val-a"]
    assert first["headline_findings"]["run1_validation_sku_count"] == 1
    assert first["collections"]["grpo_pool_cap4"]["authoritative_membership"] == {
        "sft_train": 1,
        "sft_validation": 1,
        "outside_sft_source": 0,
    }
    assert first["invariants"]["grpo_pool_has_zero_sft_validation_skus"] is False
    assert first["invariants"]["run1_has_zero_sft_validation_skus"] is False


def test_boundary_audit_detects_family_overlap_without_sku_overlap():
    rows = {
        "train-a": _row("train-a", brand="A", title="Classic Tee - Blue"),
        "val-a": _row("val-a", brand="A", title="Classic Tee - Red"),
        "probe-a": _row("probe-a", brand="P", title="Probe"),
        "eval-a": _row("eval-a", brand="E", title="Eval"),
    }
    audit = build_boundary_audit(
        collections={
            "sft_train": ["train-a"],
            "sft_validation": ["val-a"],
            "grpo_pool_cap4": ["train-a"],
            "grpo_run1_trained": ["train-a"],
            "probe_100": ["probe-a"],
            "legacy_frozen_300": ["eval-a"],
        },
        family_rows=rows,
        inputs=_inputs(),
        source_details={},
        code={},
    )

    assert audit["invariants"]["sft_train_validation_skus_disjoint"] is True
    assert audit["invariants"]["sft_train_validation_families_disjoint"] is False
    overlap = next(
        item
        for item in audit["pairwise_overlaps"]
        if item["left"] == "sft_train" and item["right"] == "sft_validation"
    )
    assert overlap["sku_overlap_count"] == 0
    assert overlap["family_overlap_count"] == 1
    assert overlap["family_overlap"] == ["a::classic tee"]
    assert overlap["left_skus_in_overlapping_families"] == ["train-a"]
    assert overlap["right_skus_in_overlapping_families"] == ["val-a"]


def test_boundary_audit_rejects_duplicate_or_unmapped_skus():
    row = _row("train-a", brand="A", title="Classic Tee")
    base = {
        "sft_train": ["train-a"],
        "sft_validation": [],
        "grpo_pool_cap4": [],
        "grpo_run1_trained": [],
        "probe_100": [],
        "legacy_frozen_300": [],
    }
    duplicated = dict(base)
    duplicated["grpo_pool_cap4"] = ["train-a", "train-a"]
    with pytest.raises(ValueError, match="duplicate SKU IDs"):
        build_boundary_audit(
            collections=duplicated,
            family_rows={"train-a": row},
            inputs=_inputs(),
            source_details={},
            code={},
        )

    unmapped = dict(base)
    unmapped["probe_100"] = ["missing"]
    with pytest.raises(ValueError, match="without product-family rows"):
        build_boundary_audit(
            collections=unmapped,
            family_rows={"train-a": row},
            inputs=_inputs(),
            source_details={},
            code={},
        )


def _write_fixture_files(tmp_path: Path) -> dict[str, Path]:
    sft_rows = [
        _row("train-a", brand="A", title="Train"),
        _row("val-a", brand="V", title="Validation"),
    ]
    sft_source = tmp_path / "sft.jsonl"
    write_jsonl(sft_rows, sft_source)
    sft_manifest = tmp_path / "sft-manifest.json"
    sft_manifest.write_text(
        json.dumps(
            {
                "version": "test-sft",
                "source": str(sft_source),
                "source_sha256": sha256_file(sft_source),
                "train": ["train-a"],
                "validation": ["val-a"],
            }
        ),
        encoding="utf-8",
    )

    pool_data = tmp_path / "pool.jsonl"
    write_jsonl([sft_rows[0]], pool_data)
    pool_manifest = tmp_path / "pool-manifest.json"
    pool_manifest.write_text(
        json.dumps(
            {
                "version": "test-pool",
                "output": {
                    "active_dataset": str(pool_data),
                    "active_dataset_sha256": sha256_file(pool_data),
                    "active_dataset_rows": 1,
                },
                "selection": {"active_skus_in_source_order": ["train-a"]},
            }
        ),
        encoding="utf-8",
    )

    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "".join(
            json.dumps({"step": 1, "sku_id": "train-a", "rollout_index": i})
            + "\n"
            for i in range(2)
        ),
        encoding="utf-8",
    )
    probe = tmp_path / "probe.jsonl"
    write_jsonl([_row("probe-a", brand="P", title="Probe", split="probe")], probe)
    legacy = tmp_path / "legacy.jsonl"
    write_jsonl([_row("eval-a", brand="E", title="Eval", split="eval")], legacy)
    return {
        "sft_manifest": sft_manifest,
        "pool_manifest": pool_manifest,
        "rollouts": rollouts,
        "probe": probe,
        "legacy": legacy,
        "sft_source": sft_source,
        "pool_data": pool_data,
    }


def test_input_loader_verifies_hashes_and_run_structure(tmp_path):
    paths = _write_fixture_files(tmp_path)
    collections, _, inputs, details = load_audit_inputs(
        repo_root=tmp_path,
        sft_manifest_path=paths["sft_manifest"],
        pool_manifest_path=paths["pool_manifest"],
        run1_rollouts_path=paths["rollouts"],
        probe_path=paths["probe"],
        legacy_eval_path=paths["legacy"],
    )

    assert collections["sft_train"] == ["train-a"]
    assert collections["grpo_run1_trained"] == ["train-a"]
    assert details["run1_rollout_structure"]["rollout_records"] == 2
    assert details["run1_rollout_structure"]["records_per_step_histogram"] == {
        "2": 1
    }
    assert inputs["sft_source"]["sha256"] == sha256_file(paths["sft_source"])

    paths["sft_source"].write_text("drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SFT manifest source hash mismatch"):
        load_audit_inputs(
            repo_root=tmp_path,
            sft_manifest_path=paths["sft_manifest"],
            pool_manifest_path=paths["pool_manifest"],
            run1_rollouts_path=paths["rollouts"],
            probe_path=paths["probe"],
            legacy_eval_path=paths["legacy"],
        )


def test_input_loader_rejects_pool_hash_drift(tmp_path):
    paths = _write_fixture_files(tmp_path)
    paths["pool_data"].write_text("drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="active-dataset hash mismatch"):
        load_audit_inputs(
            repo_root=tmp_path,
            sft_manifest_path=paths["sft_manifest"],
            pool_manifest_path=paths["pool_manifest"],
            run1_rollouts_path=paths["rollouts"],
            probe_path=paths["probe"],
            legacy_eval_path=paths["legacy"],
        )


def test_atomic_writer_refuses_to_replace_existing_output(tmp_path):
    output = tmp_path / "audit.json"
    write_exclusive_atomic_json(output, {"status": "first"})
    first_hash = hashlib.sha256(output.read_bytes()).hexdigest()

    with pytest.raises(FileExistsError, match="output already exists"):
        write_exclusive_atomic_json(output, {"status": "second"})

    assert hashlib.sha256(output.read_bytes()).hexdigest() == first_hash
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
