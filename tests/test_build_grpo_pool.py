from __future__ import annotations

import json
from collections import Counter

import pytest

from labeling.records import read_jsonl
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
