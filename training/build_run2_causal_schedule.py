#!/usr/bin/env python3
"""Build the immutable 300-product schedule shared by both Run 2 GPU arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from labeling.records import LabelStatus, Row, canonical_line, read_jsonl
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.split_sft import group_key, row_category


VERSION = "grpo-run2-causal-schedule-v1"
DEFAULT_SOURCE = "data/train_weak_grpo_cap4_sft_train_v1.jsonl"
DEFAULT_POOL_MANIFEST = "runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json"
DEFAULT_OUTPUT = "data/grpo_run2_causal_schedule_v1.jsonl"
DEFAULT_MANIFEST = "runs/grpo-run2-causal-schedule.json"
SCHEDULE_ROWS = 300
SEED = 42
NAMESPACE = "grpo-run2-causal-schedule-v1"


def _identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _set_hash(values: Sequence[str]) -> str:
    return _ordered_hash(sorted(values))


def _selection_key(row: Row) -> tuple[str, str]:
    digest = hashlib.sha256(f"{NAMESPACE}\0{SEED}\0{row.sku_id}".encode()).hexdigest()
    return digest, row.sku_id


def select_schedule(rows: Sequence[Row]) -> list[Row]:
    """Select exactly 300 unique products by a public, deterministic hash order."""
    if len(rows) != 1_438:
        raise ValueError("corrected Run 2 pool must contain exactly 1,438 rows")
    if len({row.sku_id for row in rows}) != len(rows):
        raise ValueError("corrected Run 2 pool contains duplicate SKUs")
    if any(row.split != "train" for row in rows):
        raise ValueError("causal schedule source contains a non-training row")
    if any(
        row.difficulty.sft_pass_rate is None
        or not 0.0 < row.difficulty.sft_pass_rate < 1.0
        for row in rows
    ):
        raise ValueError("causal schedule source contains an ineligible difficulty row")
    return sorted(rows, key=_selection_key)[:SCHEDULE_ROWS]


def _distribution(rows: Sequence[Row], key_fn) -> dict[str, int]:
    return dict(sorted(Counter(str(key_fn(row)) for row in rows).items()))


def _tvd(left: dict[str, int], right: dict[str, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in keys
    )


def _scorable_fields(row: Row) -> int:
    return sum(
        label.status in {LabelStatus.LABELED, LabelStatus.NOT_APPLICABLE}
        for label in row.labels.values()
    )


def _composition(rows: Sequence[Row]) -> dict[str, Any]:
    category = _distribution(rows, row_category)
    store = _distribution(rows, lambda row: row.sku_id.split(":", 2)[1])
    difficulty = _distribution(
        rows, lambda row: f"{row.difficulty.sft_pass_rate:.3f}"
    )
    families = Counter(group_key(row) for row in rows)
    scorable = [_scorable_fields(row) for row in rows]
    return {
        "rows": len(rows),
        "category_counts": category,
        "store_counts": store,
        "difficulty_counts": difficulty,
        "families": len(families),
        "maximum_rows_per_family": max(families.values()),
        "family_size_histogram": dict(
            sorted(Counter(str(value) for value in families.values()).items())
        ),
        "scorable_fields": {
            "minimum": min(scorable),
            "maximum": max(scorable),
            "mean": sum(scorable) / len(scorable),
        },
    }


def build_manifest(
    *,
    root: str | Path,
    source: str | Path,
    pool_manifest: str | Path,
    output: str | Path,
) -> tuple[list[Row], dict[str, Any]]:
    root = Path(root).resolve()
    source = (root / source).resolve()
    pool_manifest = (root / pool_manifest).resolve()
    output = (root / output).resolve()
    source_rows = read_jsonl(source)
    pool = json.loads(pool_manifest.read_text(encoding="utf-8"))
    declared = pool.get("output", {})
    if (
        declared.get("active_dataset_rows") != 1_438
        or declared.get("active_dataset_bytes") != source.stat().st_size
        or declared.get("active_dataset_sha256") != sha256_file(source)
    ):
        raise RuntimeError("corrected pool data disagrees with its manifest")
    declared_skus = pool.get("selection", {}).get("active_skus_in_source_order")
    source_skus = [row.sku_id for row in source_rows]
    if declared_skus != source_skus:
        raise RuntimeError("corrected pool source order disagrees with its manifest")

    selected = select_schedule(source_rows)
    selected_skus = [row.sku_id for row in selected]
    source_composition = _composition(source_rows)
    selected_composition = _composition(selected)
    representation = {
        "category_tvd_vs_source": _tvd(
            selected_composition["category_counts"],
            source_composition["category_counts"],
        ),
        "store_tvd_vs_source": _tvd(
            selected_composition["store_counts"],
            source_composition["store_counts"],
        ),
        "difficulty_tvd_vs_source": _tvd(
            selected_composition["difficulty_counts"],
            source_composition["difficulty_counts"],
        ),
    }
    return selected, {
        "version": VERSION,
        "status": "fixed_before_causal_gpu_dispatch",
        "selection": {
            "unit": "unique product/SKU",
            "seed": SEED,
            "namespace": NAMESPACE,
            "method": "ascending SHA-256 of namespace, seed and SKU; SKU tie-break",
            "rows": SCHEDULE_ROWS,
            "one_product_per_optimizer_step": True,
            "without_replacement": True,
            "shuffle_in_trainer": False,
            "ordered_sku_sha256": _ordered_hash(selected_skus),
            "sku_set_sha256": _set_hash(selected_skus),
            "sku_ids_in_optimizer_step_order": selected_skus,
        },
        "inputs": {
            "corrected_pool": _identity(source, root),
            "corrected_pool_manifest": _identity(pool_manifest, root),
        },
        "output": {
            "path": str(output.relative_to(root)),
            "rows": SCHEDULE_ROWS,
            "identity_pending_write": True,
        },
        "composition": {
            "source": source_composition,
            "schedule": selected_composition,
            "distance": representation,
        },
        "invariants": {
            "all_rows_from_corrected_pool": True,
            "all_rows_authoritative_training_split": True,
            "all_rows_have_nonterminal_starting_sft_pass_rate": True,
            "duplicate_skus": 0,
            "maximum_rows_per_family_at_most_four": (
                selected_composition["maximum_rows_per_family"] <= 4
            ),
            "validation_or_confirmation_rows_used": False,
        },
    }


def _write_exclusive_jsonl(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"schedule output already exists: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_line(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--pool-manifest", default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    output = (root / args.output).resolve()
    manifest_path = (root / args.manifest).resolve()
    rows, manifest = build_manifest(
        root=root,
        source=args.source,
        pool_manifest=args.pool_manifest,
        output=args.output,
    )
    _write_exclusive_jsonl(output, rows)
    manifest["output"] = {
        **_identity(output, root),
        "rows": len(rows),
        "identity_pending_write": False,
    }
    write_exclusive_atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
