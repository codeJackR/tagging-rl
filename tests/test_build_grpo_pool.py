from __future__ import annotations

import json
from collections import Counter, defaultdict

import pytest

from labeling.records import read_jsonl
import training.build_grpo_pool as build_grpo_pool
from training.audit_grpo_pool import select_deterministic_family_cap
from training.build_grpo_pool import select_pool, write_pool_artifacts
from training.dataset import load_grpo_prompts
from training.split_sft import group_key
from verifier import load_pack


def test_cap_selector_is_deterministic_and_preserves_source_order():
    rows = read_jsonl("data/train_weak_sft_scored.jsonl")
    eligible = [row for row in rows if 0 < row.difficulty.sft_pass_rate < 1]
    first = select_deterministic_family_cap(eligible, cap=4, seed=42)
    second = select_deterministic_family_cap(eligible, cap=4, seed=42)

    assert [row.sku_id for row in first] == [row.sku_id for row in second]
    source_positions = {row.sku_id: index for index, row in enumerate(rows)}
    assert [source_positions[row.sku_id] for row in first] == sorted(
        source_positions[row.sku_id] for row in first
    )
    assert len(first) == 1565
    assert max(Counter(group_key(row) for row in first).values()) == 4


def test_real_pool_selection_matches_audited_cap_four():
    rows = read_jsonl("data/train_weak_sft_scored.jsonl")
    audit = json.load(open("runs/sft-difficulty-k8/retained-pool-audit.json"))
    eligible, active, capped, scenario = select_pool(
        rows, audit, cap=4, seed=42
    )

    assert len(eligible) == 1702
    assert len(active) == 1565
    assert len(capped) == 137
    assert scenario["active_sku_set_sha256"] == (
        "77d926c88447e3cda5852f015629a4eae8bb9c7e32a00a67694abd985fb75c76"
    )
    assert {group_key(row) for row in active} == {
        group_key(row) for row in eligible
    }


def test_authoritative_sft_manifest_overrides_embedded_train_split():
    rows = read_jsonl("data/train_weak_sft_scored.jsonl")
    split_manifest = json.load(open("data/splits/sft-v1.json"))

    # The W1-level split calls all 3,600 rows "train". The later SFT manifest
    # divides that corpus into 3,240 training and 360 validation SKUs, so the
    # embedded value cannot decide whether a row is eligible for GRPO.
    assert len(rows) == 3600
    assert all(row.split == "train" for row in rows)

    selected = build_grpo_pool.filter_authoritative_training_rows(
        rows,
        split_manifest_path="data/splits/sft-v1.json",
    )

    assert [row.sku_id for row in selected] == [
        row.sku_id
        for row in rows
        if row.sku_id in set(split_manifest["train"])
    ]
    assert len(selected) == 3240
    assert not ({row.sku_id for row in selected} & set(split_manifest["validation"]))


def test_written_pool_excludes_authoritative_sft_validation_rows(tmp_path):
    output_data = tmp_path / "train-only-active.jsonl"
    output_manifest = tmp_path / "train-only-manifest.json"
    split_manifest = json.load(open("data/splits/sft-v1.json"))

    manifest = write_pool_artifacts(
        scored_path="data/train_weak_sft_scored.jsonl",
        audit_path="runs/sft-difficulty-k8/retained-pool-audit.json",
        split_manifest_path="data/splits/sft-v1.json",
        output_data_path=output_data,
        output_manifest_path=output_manifest,
        cap=4,
        seed=42,
        created_at_utc="2026-08-11T00:00:00+00:00",
    )

    output_rows = read_jsonl(output_data)
    output_ids = {row.sku_id for row in output_rows}
    validation_ids = set(split_manifest["validation"])

    assert len(output_rows) == 1438
    assert manifest["version"] == "grpo-pool-cap4-sft-train-v1"
    assert manifest["selection"]["active_rows"] == 1438
    assert not (output_ids & validation_ids)
    assert max(Counter(group_key(row) for row in output_rows).values()) <= 4
    assert manifest["inputs"]["sft_split_manifest"] == "data/splits/sft-v1.json"
    assert all(manifest["invariants"].values())


def test_cli_forwards_optional_authoritative_sft_manifest(monkeypatch):
    captured = {}

    def fake_write_pool_artifacts(**kwargs):
        captured.update(kwargs)
        return {
            "selection": {
                "eligible_rows": 1565,
                "active_rows": 1438,
                "capped_eligible_rows": 127,
            }
        }

    monkeypatch.setattr(
        build_grpo_pool,
        "write_pool_artifacts",
        fake_write_pool_artifacts,
    )

    assert build_grpo_pool.parse_args([]).sft_split_manifest is None
    result = build_grpo_pool.main(
        [
            "--sft-split-manifest",
            "data/splits/sft-v1.json",
            "--output-data",
            "unused-data.jsonl",
            "--output-manifest",
            "unused-manifest.json",
        ]
    )

    assert result == 0
    assert captured["split_manifest_path"] == "data/splits/sft-v1.json"


def test_corrected_pool_fails_closed_on_sft_manifest_source_hash_drift(tmp_path):
    split_manifest = json.load(open("data/splits/sft-v1.json"))
    split_manifest["source_sha256"] = "0" * 64
    tampered_manifest = tmp_path / "tampered-sft-split.json"
    tampered_manifest.write_text(
        json.dumps(split_manifest),
        encoding="utf-8",
    )
    output_data = tmp_path / "must-not-exist.jsonl"
    output_manifest = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="SFT split manifest source hash mismatch"):
        write_pool_artifacts(
            scored_path="data/train_weak_sft_scored.jsonl",
            audit_path="runs/sft-difficulty-k8/retained-pool-audit.json",
            split_manifest_path=tampered_manifest,
            output_data_path=output_data,
            output_manifest_path=output_manifest,
            cap=4,
            seed=42,
            created_at_utc="2026-08-11T00:00:00+00:00",
        )

    assert not output_data.exists()
    assert not output_manifest.exists()


def test_authoritative_filter_rejects_train_validation_family_overlap(tmp_path):
    rows = read_jsonl("data/train_weak_sft_scored.jsonl")
    split_manifest = json.load(open("data/splits/sft-v1.json"))
    training_ids = set(split_manifest["train"])
    training_families = defaultdict(list)
    for row in rows:
        if row.sku_id in training_ids:
            training_families[group_key(row)].append(row.sku_id)

    shared_family = next(
        sorted(members)
        for _, members in sorted(training_families.items())
        if len(members) >= 2
    )
    moved_to_validation = shared_family[1]
    split_manifest["train"].remove(moved_to_validation)
    split_manifest["validation"].append(moved_to_validation)
    family_leaking_manifest = tmp_path / "family-leaking-sft-split.json"
    family_leaking_manifest.write_text(
        json.dumps(split_manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SFT split manifest has family overlap"):
        build_grpo_pool.filter_authoritative_training_rows(
            rows,
            split_manifest_path=family_leaking_manifest,
        )


def test_written_pool_manifest_and_loader_handoff(tmp_path):
    pack = load_pack("packs/vastraa_taste_v1")
    output_data = tmp_path / "active.jsonl"
    output_manifest = tmp_path / "manifest.json"
    manifest = write_pool_artifacts(
        scored_path="data/train_weak_sft_scored.jsonl",
        audit_path="runs/sft-difficulty-k8/retained-pool-audit.json",
        output_data_path=output_data,
        output_manifest_path=output_manifest,
        cap=4,
        seed=42,
        created_at_utc="2026-08-02T00:00:00+00:00",
    )

    assert manifest["version"] == "grpo-pool-cap4-v1"
    assert manifest["selection"]["active_rows"] == 1565
    assert manifest["selection"]["capped_eligible_rows"] == 137
    assert len(manifest["selection"]["active_skus_in_source_order"]) == 1565
    assert len(manifest["selection"]["capped_eligible_skus_in_source_order"]) == 137
    assert all(manifest["invariants"].values())
    assert manifest["code"]["implementation_file"] == "training/build_grpo_pool.py"
    assert output_manifest.is_file()

    dataset = load_grpo_prompts(
        pack, path=output_data, require_pass_rate_band=True
    )
    assert len(dataset) == 1565
    assert "completion" not in dataset[0]

    with pytest.raises(FileExistsError, match="output already exists"):
        write_pool_artifacts(
            scored_path="data/train_weak_sft_scored.jsonl",
            audit_path="runs/sft-difficulty-k8/retained-pool-audit.json",
            output_data_path=output_data,
            output_manifest_path=output_manifest,
            cap=4,
            seed=42,
            created_at_utc="2026-08-02T00:00:00+00:00",
        )
