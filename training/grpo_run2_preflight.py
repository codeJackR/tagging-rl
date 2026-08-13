#!/usr/bin/env python3
"""CPU-only launch boundary for the corrected GRPO Run 2 prompt pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from labeling.records import read_jsonl
from training.audit_data_boundaries import sha256_file, sku_set_sha256
from training.split_sft import group_key

PREFLIGHT_VERSION = "grpo-run2-pool-preflight-v1"
POOL_VERSION = "grpo-pool-cap4-sft-train-v1"
DEFAULT_DATA = "data/train_weak_grpo_cap4_sft_train_v1.jsonl"
DEFAULT_MANIFEST = (
    "runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json"
)
DEFAULT_SFT_SPLIT_MANIFEST = "data/splits/sft-v1.json"
LOCKED_DATA_SHA256 = (
    "1ca64f668b0359c2e83850832d5db2ffaf5a5f621556ac4776cf9d5c3fb26a53"
)
LOCKED_MANIFEST_SHA256 = (
    "42ca7b3ad0b1a1e61539493a693b33ea56238f30004e6a25bcdb9bd24e19282a"
)
LOCKED_SFT_SPLIT_MANIFEST_SHA256 = (
    "4d14d46fa4f7df95a24658c741940db64093e7798b5ccd1558f4faa29bbe9a3b"
)
LOCKED_ROWS = 1_438
LOCKED_DATA_BYTES = 3_191_282
LOCKED_FAMILIES = 1_051


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return value


def verify_run2_pool(
    *,
    repo_root: str | Path,
    data_path: str | Path = DEFAULT_DATA,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    split_manifest_path: str | Path = DEFAULT_SFT_SPLIT_MANIFEST,
    expected_data_sha256: str = LOCKED_DATA_SHA256,
    expected_manifest_sha256: str = LOCKED_MANIFEST_SHA256,
    expected_split_manifest_sha256: str = LOCKED_SFT_SPLIT_MANIFEST_SHA256,
    expected_rows: int = LOCKED_ROWS,
    expected_bytes: int = LOCKED_DATA_BYTES,
    expected_families: int = LOCKED_FAMILIES,
) -> dict:
    """Independently verify the corrected pool before any GPU work."""
    repo_root = Path(repo_root).resolve()
    data_path = _resolve(repo_root, data_path)
    manifest_path = _resolve(repo_root, manifest_path)
    split_manifest_path = _resolve(repo_root, split_manifest_path)

    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise RuntimeError("locked Run 2 pool manifest checksum mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("version") != POOL_VERSION:
        raise RuntimeError("unexpected Run 2 pool manifest version")

    invariants = manifest.get("invariants")
    required_invariants = {
        "all_active_rows_are_authoritative_sft_train",
        "active_and_sft_validation_skus_are_disjoint",
        "active_and_sft_validation_families_are_disjoint",
        "family_cap_respected",
    }
    if (
        not isinstance(invariants, dict)
        or not required_invariants <= set(invariants)
        or not all(invariants.values())
    ):
        raise RuntimeError("Run 2 pool manifest contains failed or missing invariants")

    inputs = manifest.get("inputs", {})
    recorded_split_path = _resolve(
        repo_root, inputs.get("sft_split_manifest", "")
    )
    if recorded_split_path != split_manifest_path:
        raise RuntimeError("Run 2 pool references a different SFT split manifest")
    actual_split_sha = sha256_file(split_manifest_path)
    if actual_split_sha != expected_split_manifest_sha256:
        raise RuntimeError("locked SFT split manifest checksum mismatch")
    if actual_split_sha != inputs.get("sft_split_manifest_sha256"):
        raise RuntimeError("Run 2 pool SFT split-manifest lineage mismatch")

    split_manifest = _read_json(split_manifest_path)
    if split_manifest.get("version") != "sft-v1":
        raise RuntimeError("unexpected SFT split manifest version")
    split_source_path = _resolve(repo_root, split_manifest.get("source", ""))
    if sha256_file(split_source_path) != split_manifest.get("source_sha256"):
        raise RuntimeError("SFT split source checksum mismatch")
    split_source_rows = read_jsonl(split_source_path)
    source_by_sku = {row.sku_id: row for row in split_source_rows}
    if len(source_by_sku) != len(split_source_rows):
        raise RuntimeError("SFT split source contains duplicate SKUs")

    train_ids = split_manifest.get("train")
    validation_ids = split_manifest.get("validation")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise RuntimeError("SFT split assignments are not lists")
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(
        validation_ids
    ):
        raise RuntimeError("SFT split assignments contain duplicate SKUs")
    train_set = set(train_ids)
    validation_set = set(validation_ids)
    if train_set & validation_set:
        raise RuntimeError("SFT train and validation SKU sets overlap")
    if train_set | validation_set != set(source_by_sku):
        raise RuntimeError("SFT split assignments do not cover their source")
    train_families = {group_key(source_by_sku[sku]) for sku in train_set}
    validation_families = {
        group_key(source_by_sku[sku]) for sku in validation_set
    }
    if train_families & validation_families:
        raise RuntimeError("SFT train and validation families overlap")

    output = manifest.get("output", {})
    if _resolve(repo_root, output.get("active_dataset", "")) != data_path:
        raise RuntimeError("Run 2 data path disagrees with pool manifest")
    actual_data_sha = sha256_file(data_path)
    if actual_data_sha != expected_data_sha256:
        raise RuntimeError("locked Run 2 pool data checksum mismatch")
    if actual_data_sha != output.get("active_dataset_sha256"):
        raise RuntimeError("Run 2 data checksum disagrees with pool manifest")
    if data_path.stat().st_size != expected_bytes:
        raise RuntimeError("locked Run 2 pool byte size mismatch")
    if data_path.stat().st_size != output.get("active_dataset_bytes"):
        raise RuntimeError("Run 2 pool byte size disagrees with manifest")

    rows = read_jsonl(data_path)
    if len(rows) != expected_rows or len(rows) != output.get("active_dataset_rows"):
        raise RuntimeError("Run 2 pool row count mismatch")
    sku_ids = [row.sku_id for row in rows]
    if len(set(sku_ids)) != len(sku_ids):
        raise RuntimeError("Run 2 pool contains duplicate SKUs")
    selection = manifest.get("selection", {})
    if sku_ids != selection.get("active_skus_in_source_order"):
        raise RuntimeError("Run 2 pool SKU order disagrees with manifest")
    if sku_set_sha256(sku_ids) != selection.get("active_sku_set_sha256"):
        raise RuntimeError("Run 2 pool SKU set disagrees with manifest")

    active_ids = set(sku_ids)
    if not active_ids <= train_set:
        raise RuntimeError("Run 2 pool contains a non-training SKU")
    if active_ids & validation_set:
        raise RuntimeError("Run 2 pool contains an SFT-validation SKU")
    active_families = {group_key(source_by_sku[sku]) for sku in active_ids}
    if active_families & validation_families:
        raise RuntimeError("Run 2 pool contains an SFT-validation family")
    if len(active_families) != expected_families:
        raise RuntimeError("Run 2 pool family count mismatch")

    policy = manifest.get("policy", {})
    if (
        policy.get("family_cap") != 4
        or policy.get("selection_seed") != 42
        or policy.get("eligibility_rule") != "0 < sft_pass_rate < 1"
    ):
        raise RuntimeError("Run 2 pool policy drifted")
    family_counts = Counter(group_key(source_by_sku[sku]) for sku in active_ids)
    if max(family_counts.values()) > policy["family_cap"]:
        raise RuntimeError("Run 2 pool violates its family cap")
    if any(
        row.difficulty.sft_pass_rate is None
        or not 0 < row.difficulty.sft_pass_rate < 1
        for row in rows
    ):
        raise RuntimeError("Run 2 pool contains a difficulty-ineligible row")

    return {
        "version": PREFLIGHT_VERSION,
        "status": "passed",
        "cuda_imports_performed": False,
        "pool": {
            "data_path": str(data_path),
            "data_sha256": actual_data_sha,
            "data_bytes": data_path.stat().st_size,
            "manifest_path": str(manifest_path),
            "manifest_sha256": actual_manifest_sha,
            "manifest_version": manifest["version"],
            "rows": len(rows),
            "families": len(active_families),
            "maximum_family_size": max(family_counts.values()),
            "active_sku_set_sha256": selection["active_sku_set_sha256"],
        },
        "split_authority": {
            "manifest_path": str(split_manifest_path),
            "manifest_sha256": actual_split_sha,
            "manifest_version": split_manifest["version"],
            "source_path": str(split_source_path),
            "source_sha256": split_manifest["source_sha256"],
            "training_skus": len(train_set),
            "validation_skus": len(validation_set),
            "active_validation_sku_overlap": 0,
            "active_validation_family_overlap": 0,
        },
        "manifest_invariants": dict(invariants),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--sft-split-manifest", default=DEFAULT_SFT_SPLIT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_run2_pool(
        repo_root=args.repo_root,
        data_path=args.data,
        manifest_path=args.manifest,
        split_manifest_path=args.sft_split_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
