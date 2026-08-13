from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from labeling.records import read_jsonl
from training.audit_data_boundaries import sha256_file
from training.grpo_run2_preflight import (
    DEFAULT_DATA,
    DEFAULT_MANIFEST,
    DEFAULT_SFT_SPLIT_MANIFEST,
    LOCKED_DATA_SHA256,
    LOCKED_MANIFEST_SHA256,
    LOCKED_SFT_SPLIT_MANIFEST_SHA256,
    verify_run2_pool,
)
from training.split_sft import group_key

ROOT = Path(__file__).resolve().parent.parent


def test_real_corrected_run2_pool_passes_independent_preflight():
    report = verify_run2_pool(repo_root=ROOT)

    assert report["status"] == "passed"
    assert report["cuda_imports_performed"] is False
    assert report["pool"]["rows"] == 1438
    assert report["pool"]["families"] == 1051
    assert report["pool"]["maximum_family_size"] == 4
    assert report["split_authority"]["training_skus"] == 3240
    assert report["split_authority"]["validation_skus"] == 360
    assert report["split_authority"]["active_validation_sku_overlap"] == 0
    assert report["split_authority"]["active_validation_family_overlap"] == 0


def test_run2_preflight_rejects_wrong_manifest_version(tmp_path):
    manifest = json.load(open(ROOT / DEFAULT_MANIFEST))
    manifest["version"] = "historical-run1-version"
    changed = tmp_path / "wrong-version.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected Run 2 pool manifest version"):
        verify_run2_pool(
            repo_root=ROOT,
            manifest_path=changed,
            expected_manifest_sha256=sha256_file(changed),
        )


def test_run2_preflight_rejects_failed_builder_invariant(tmp_path):
    manifest = json.load(open(ROOT / DEFAULT_MANIFEST))
    manifest["invariants"]["active_and_sft_validation_skus_are_disjoint"] = False
    changed = tmp_path / "failed-invariant.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed or missing invariants"):
        verify_run2_pool(
            repo_root=ROOT,
            manifest_path=changed,
            expected_manifest_sha256=sha256_file(changed),
        )


def test_run2_preflight_independently_rejects_family_leakage(tmp_path):
    split_manifest = json.load(open(ROOT / DEFAULT_SFT_SPLIT_MANIFEST))
    rows = read_jsonl(ROOT / split_manifest["source"])
    active_ids = {
        row.sku_id for row in read_jsonl(ROOT / DEFAULT_DATA)
    }
    training_ids = set(split_manifest["train"])
    families = defaultdict(list)
    for row in rows:
        if row.sku_id in training_ids:
            families[group_key(row)].append(row.sku_id)
    shared_family = next(
        members
        for _, members in sorted(families.items())
        if any(sku in active_ids for sku in members)
        and any(sku not in active_ids for sku in members)
    )
    moved = next(sku for sku in shared_family if sku not in active_ids)
    split_manifest["train"].remove(moved)
    split_manifest["validation"].append(moved)
    changed_split = tmp_path / "family-leaking-sft.json"
    changed_split.write_text(json.dumps(split_manifest), encoding="utf-8")

    pool_manifest = json.load(open(ROOT / DEFAULT_MANIFEST))
    pool_manifest["inputs"]["sft_split_manifest"] = str(changed_split)
    pool_manifest["inputs"]["sft_split_manifest_sha256"] = sha256_file(
        changed_split
    )
    changed_pool = tmp_path / "changed-pool-manifest.json"
    changed_pool.write_text(json.dumps(pool_manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SFT train and validation families overlap"):
        verify_run2_pool(
            repo_root=ROOT,
            manifest_path=changed_pool,
            split_manifest_path=changed_split,
            expected_manifest_sha256=sha256_file(changed_pool),
            expected_split_manifest_sha256=sha256_file(changed_split),
        )


def test_run2_preflight_lock_constants_match_real_artifacts():
    assert sha256_file(ROOT / DEFAULT_DATA) == LOCKED_DATA_SHA256
    assert sha256_file(ROOT / DEFAULT_MANIFEST) == LOCKED_MANIFEST_SHA256
    assert (
        sha256_file(ROOT / DEFAULT_SFT_SPLIT_MANIFEST)
        == LOCKED_SFT_SPLIT_MANIFEST_SHA256
    )
