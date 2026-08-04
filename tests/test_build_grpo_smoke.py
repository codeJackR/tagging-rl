from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from labeling.records import read_jsonl
from training.build_grpo_smoke import (
    select_smoke_rows,
    verify_active_handoff,
    write_smoke_artifacts,
)
from training.dataset import load_grpo_prompts
from training.split_sft import group_key
from verifier import load_pack

ACTIVE = Path("data/train_weak_grpo_cap4.jsonl")
POOL_MANIFEST = Path("runs/sft-difficulty-k8/grpo-pool-cap4-manifest.json")
EXPECTED_SKUS = [
    "shopify:www.tentree.com:8106124673210",
    "shopify:fahertybrand.com:8164246683717",
    "shopify:fahertybrand.com:7552625803333",
    "shopify:www.rothys.com:7543272243294",
    "shopify:www.outdoorvoices.com:7686276677710",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_smoke_selection_is_deterministic_and_family_distinct():
    rows = read_jsonl(ACTIVE)
    first_candidates, first = select_smoke_rows(rows)
    second_candidates, second = select_smoke_rows(rows)

    assert len(first_candidates) == len(second_candidates) == 176
    assert len({group_key(row) for row in first_candidates}) == 160
    assert [row.sku_id for row in first] == EXPECTED_SKUS
    assert [row.sku_id for row in second] == EXPECTED_SKUS
    assert len({group_key(row) for row in first}) == 5
    assert all(row.difficulty.sft_pass_rate == 0.5 for row in first)


def test_active_handoff_is_verified_before_selection():
    rows = read_jsonl(ACTIVE)
    manifest = json.loads(POOL_MANIFEST.read_text(encoding="utf-8"))
    verify_active_handoff(rows, manifest, ACTIVE)

    broken_manifest = json.loads(json.dumps(manifest))
    broken_manifest["output"]["active_dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash disagrees"):
        verify_active_handoff(rows, broken_manifest, ACTIVE)


def test_committed_smoke_artifacts_are_locked():
    data_path = Path("data/train_weak_grpo_smoke_v1.jsonl")
    manifest_path = Path("data/splits/grpo-smoke-v1.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert sha256_file(data_path) == (
        "268373ceb08c53125976493340d972a47c90e10911e919002716590f75ca4084"
    )
    assert sha256_file(manifest_path) == (
        "e898510534d967b9a35367e0aba5a564e6cb564e2c326d45804d0319d528dd05"
    )
    assert manifest["output"]["smoke_dataset_sha256"] == sha256_file(data_path)
    assert manifest["selection"]["selected_sku_order_sha256"] == (
        "99820aef9777190af82b999d587a260198e65ef9c420c0ca5d6befde06fe7af0"
    )
    assert manifest["selection"]["selected_skus_in_step_order"] == EXPECTED_SKUS
    assert manifest["code"]["implementation_file_sha256"] == (
        "55e08782c123e623546ae53dd04d665730046dde55a7cd303d021e162c68882d"
    )


def test_written_smoke_manifest_and_prompt_handoff(tmp_path):
    output_data = tmp_path / "smoke.jsonl"
    output_manifest = tmp_path / "manifest.json"
    manifest = write_smoke_artifacts(
        active_path=ACTIVE,
        pool_manifest_path=POOL_MANIFEST,
        output_data_path=output_data,
        output_manifest_path=output_manifest,
        seed=42,
        target_pass_rate=0.5,
        count=5,
        created_at_utc="2026-08-04T00:00:00+00:00",
    )

    assert manifest["version"] == "grpo-smoke-v1"
    assert manifest["selection"]["candidate_rows"] == 176
    assert manifest["selection"]["candidate_families"] == 160
    assert manifest["selection"]["selected_skus_in_step_order"] == EXPECTED_SKUS
    assert manifest["selection"]["selected_rows"] == 5
    assert manifest["selection"]["selected_families"] == 5
    assert manifest["output"]["smoke_dataset_rows"] == 5
    assert all(manifest["invariants"].values())
    assert manifest["code"]["implementation_file"] == (
        "training/build_grpo_smoke.py"
    )
    assert output_manifest.is_file()

    output_rows = read_jsonl(output_data)
    assert [row.sku_id for row in output_rows] == EXPECTED_SKUS
    pack = load_pack("packs/vastraa_taste_v1")
    dataset = load_grpo_prompts(
        pack, path=output_data, require_pass_rate_band=True
    )
    assert len(dataset) == 5
    assert dataset["sku_id"] == EXPECTED_SKUS
    assert "completion" not in dataset.column_names
    assert all(json.loads(gold) for gold in dataset["gold"])

    with pytest.raises(FileExistsError, match="output already exists"):
        write_smoke_artifacts(
            active_path=ACTIVE,
            pool_manifest_path=POOL_MANIFEST,
            output_data_path=output_data,
            output_manifest_path=output_manifest,
            seed=42,
            target_pass_rate=0.5,
            count=5,
            created_at_utc="2026-08-04T00:00:00+00:00",
        )


def test_selection_rejects_impossible_or_invalid_requests():
    rows = read_jsonl(ACTIVE)

    with pytest.raises(ValueError, match="positive"):
        select_smoke_rows(rows, count=0)
    with pytest.raises(ValueError, match="strictly between"):
        select_smoke_rows(rows, target_pass_rate=1.0)
    with pytest.raises(ValueError, match="distinct families"):
        select_smoke_rows(rows, target_pass_rate=0.5, count=161)
