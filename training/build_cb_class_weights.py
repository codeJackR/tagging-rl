#!/usr/bin/env python3
"""Build Candidate CB's immutable training-only class-support weight map.

This module derives support exclusively from the corrected 1,438-row active
GRPO training pool.  It does not parse completions, calculate candidate rewards,
read rollout artifacts, or inspect validation/frozen data.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from labeling.records import AttributeLabel, LabelStatus, Row, read_jsonl
from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.grpo_run2_preflight import (
    DEFAULT_DATA,
    DEFAULT_MANIFEST,
    DEFAULT_SFT_SPLIT_MANIFEST,
    verify_run2_pool,
)
from training.reward_scale_contract import (
    CLASS_WEIGHT_MAX,
    CLASS_WEIGHT_MIN,
    KNOWN_CELLS,
    UNKNOWN_CELLS,
    class_weight,
)
from verifier import Pack, load_pack


VERSION = "grpo-run2-cb-class-weights-v1"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_OUTPUT = "runs/grpo-run2-cb-class-weights.json"
NOT_APPLICABLE_CLASS = "__not_applicable__"
EXPECTED_ROWS = 1_438
EXPECTED_FIELDS = 15
EXPECTED_ATTRIBUTE_CLASS_PAIRS = 116
EXPECTED_CLASSES_BELOW_FIVE = 17
EXPECTED_CLASSES_BELOW_TEN = 30


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _file_metadata(path: Path, *, repo_root: Path) -> dict[str, Any]:
    try:
        display_path = str(path.relative_to(repo_root))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def label_class_keys(
    *, field_name: str, label: AttributeLabel, pack: Pack
) -> tuple[str, ...]:
    """Return support keys for one gold label, excluding gold unknown."""
    if field_name not in pack.specs:
        raise ValueError(f"unknown pack field: {field_name}")
    spec = pack.specs[field_name]
    if label.status is LabelStatus.UNKNOWN:
        return ()
    if label.status is LabelStatus.NOT_APPLICABLE:
        return (NOT_APPLICABLE_CLASS,)

    value = label.value
    if spec.kind == "multi":
        if not isinstance(value, list) or not value:
            raise ValueError(f"{field_name}: labeled multi field must contain values")
        if len(value) != len(set(value)):
            raise ValueError(f"{field_name}: labeled multi field contains duplicates")
        values = tuple(value)
    else:
        if not isinstance(value, str):
            raise ValueError(f"{field_name}: labeled scalar field must be a string")
        values = (value,)
    invalid = sorted(set(values) - set(spec.values))
    if invalid:
        raise ValueError(f"{field_name}: labels outside controlled vocabulary: {invalid}")
    return values


def derive_class_weight_map(rows: Sequence[Row], pack: Pack) -> dict[str, Any]:
    """Aggregate support at attribute/class grain and derive bounded weights."""
    if not rows:
        raise ValueError("class-weight source rows cannot be empty")
    sku_ids = [row.sku_id for row in rows]
    if len(set(sku_ids)) != len(sku_ids):
        raise ValueError("class-weight source contains duplicate SKUs")

    support_by_field = {field_name: Counter() for field_name in pack.field_names}
    known_cells = 0
    unknown_cells = 0
    not_applicable_cells = 0
    class_observations = 0
    multi_label_extra_observations = 0

    for row in rows:
        if set(row.labels) != set(pack.field_names):
            missing = sorted(set(pack.field_names) - set(row.labels))
            extra = sorted(set(row.labels) - set(pack.field_names))
            raise ValueError(
                f"{row.sku_id}: labels differ from pack fields; "
                f"missing={missing}, extra={extra}"
            )
        for field_name in pack.field_names:
            label = row.labels[field_name]
            keys = label_class_keys(field_name=field_name, label=label, pack=pack)
            if label.status is LabelStatus.UNKNOWN:
                unknown_cells += 1
                if keys:
                    raise RuntimeError("gold unknown unexpectedly produced class support")
                continue
            known_cells += 1
            if label.status is LabelStatus.NOT_APPLICABLE:
                not_applicable_cells += 1
            support_by_field[field_name].update(keys)
            class_observations += len(keys)
            multi_label_extra_observations += max(0, len(keys) - 1)

    attributes: dict[str, Any] = {}
    all_supports: list[int] = []
    all_weights: list[float] = []
    clipping_counts = {"minimum": 0, "maximum": 0, "unclipped": 0}
    not_applicable_fields: list[str] = []
    for field_name in pack.field_names:
        counter = support_by_field[field_name]
        if not counter:
            raise ValueError(f"{field_name}: no positive training support")
        supports = list(counter.values())
        if any(value <= 0 for value in supports):
            raise RuntimeError(f"{field_name}: nonpositive class support")
        median_support = statistics.median(supports)
        classes: dict[str, Any] = {}
        for class_name in sorted(counter):
            support = counter[class_name]
            raw_weight = math.sqrt(median_support / support)
            weight = class_weight(support, median_support)
            clipped_at = (
                "minimum"
                if raw_weight < CLASS_WEIGHT_MIN
                else "maximum"
                if raw_weight > CLASS_WEIGHT_MAX
                else None
            )
            clipping_counts[clipped_at or "unclipped"] += 1
            classes[class_name] = {
                "support": support,
                "raw_weight": raw_weight,
                "weight": weight,
                "clipped_at": clipped_at,
            }
            all_supports.append(support)
            all_weights.append(weight)
        if NOT_APPLICABLE_CLASS in classes:
            not_applicable_fields.append(field_name)
        attributes[field_name] = {
            "observed_classes": len(classes),
            "median_positive_support": median_support,
            "classes": classes,
        }

    return {
        "rows": len(rows),
        "fields": len(pack.field_names),
        "known_field_cells": known_cells,
        "unknown_field_cells": unknown_cells,
        "not_applicable_field_cells": not_applicable_cells,
        "class_observations": class_observations,
        "multi_label_extra_observations": multi_label_extra_observations,
        "observed_attribute_class_pairs": len(all_supports),
        "classes_below_five": sum(value < 5 for value in all_supports),
        "classes_below_ten": sum(value < 10 for value in all_supports),
        "minimum_support": min(all_supports),
        "maximum_support": max(all_supports),
        "minimum_derived_weight": min(all_weights),
        "maximum_derived_weight": max(all_weights),
        "clipping_counts": clipping_counts,
        "not_applicable_fields": not_applicable_fields,
        "attributes": attributes,
    }


def _validate_locked_counts(weight_map: Mapping[str, Any]) -> dict[str, bool]:
    invariants = {
        "active_row_count_matches": weight_map["rows"] == EXPECTED_ROWS,
        "field_count_matches": weight_map["fields"] == EXPECTED_FIELDS,
        "known_cell_count_matches": weight_map["known_field_cells"] == KNOWN_CELLS,
        "unknown_cell_count_matches": weight_map["unknown_field_cells"] == UNKNOWN_CELLS,
        "attribute_class_pair_count_matches": (
            weight_map["observed_attribute_class_pairs"]
            == EXPECTED_ATTRIBUTE_CLASS_PAIRS
        ),
        "rare_below_five_count_matches": (
            weight_map["classes_below_five"] == EXPECTED_CLASSES_BELOW_FIVE
        ),
        "rare_below_ten_count_matches": (
            weight_map["classes_below_ten"] == EXPECTED_CLASSES_BELOW_TEN
        ),
        "unknown_token_has_no_support": all(
            "unknown" not in attribute["classes"]
            for attribute in weight_map["attributes"].values()
        ),
        "not_applicable_is_explicit_where_observed": bool(
            weight_map["not_applicable_fields"]
        ),
        "all_weights_within_locked_bounds": (
            CLASS_WEIGHT_MIN <= weight_map["minimum_derived_weight"]
            and weight_map["maximum_derived_weight"] <= CLASS_WEIGHT_MAX
        ),
        "all_class_pairs_have_one_clipping_state": (
            sum(weight_map["clipping_counts"].values())
            == weight_map["observed_attribute_class_pairs"]
        ),
        "multi_label_observation_accounting_matches": (
            weight_map["class_observations"]
            == weight_map["known_field_cells"]
            + weight_map["multi_label_extra_observations"]
        ),
    }
    failed = sorted(name for name, passed in invariants.items() if not passed)
    if failed:
        raise RuntimeError(f"CB class-weight invariants failed: {failed}")
    return invariants


def build_cb_class_weight_artifact(
    *,
    repo_root: str | Path,
    data_path: str | Path = DEFAULT_DATA,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    split_manifest_path: str | Path = DEFAULT_SFT_SPLIT_MANIFEST,
    pack_path: str | Path = DEFAULT_PACK,
) -> dict[str, Any]:
    """Verify lineage, derive the map, and return one deterministic artifact."""
    repo_root = Path(repo_root).resolve()
    data_path = _resolve(repo_root, data_path)
    manifest_path = _resolve(repo_root, manifest_path)
    split_manifest_path = _resolve(repo_root, split_manifest_path)
    pack_path = _resolve(repo_root, pack_path)

    preflight = verify_run2_pool(
        repo_root=repo_root,
        data_path=data_path,
        manifest_path=manifest_path,
        split_manifest_path=split_manifest_path,
    )
    rows = read_jsonl(data_path)
    pack = load_pack(pack_path)
    weight_map = derive_class_weight_map(rows, pack)
    invariants = _validate_locked_counts(weight_map)
    implementation_path = Path(__file__).resolve()
    return {
        "version": VERSION,
        "status": "passed",
        "role": "immutable Candidate CB class-support and bounded-weight lookup",
        "selection_boundary": {
            "allowed": "corrected active SFT-training pool gold labels only",
            "prohibited": [
                "rollout completions",
                "candidate reward outcomes",
                "SFT validation",
                "legacy frozen 300",
                "probe 100",
                "future confirmation data",
            ],
            "candidate_completion_rewards_calculated": False,
        },
        "cuda_imports_performed": False,
        "inputs": {
            "active_pool": _file_metadata(data_path, repo_root=repo_root),
            "active_pool_manifest": _file_metadata(manifest_path, repo_root=repo_root),
            "sft_split_manifest": _file_metadata(
                split_manifest_path, repo_root=repo_root
            ),
            "pack_vocab": _file_metadata(pack_path / "vocab.yaml", repo_root=repo_root),
            "pool_preflight": preflight,
        },
        "implementation": _file_metadata(implementation_path, repo_root=repo_root),
        "weight_contract": {
            "formula": "clip(sqrt(attribute_median_positive_support / class_support), 0.5, 2.0)",
            "minimum": CLASS_WEIGHT_MIN,
            "maximum": CLASS_WEIGHT_MAX,
            "support_grain": "one count per gold class within attribute",
            "unknown_handling": "excluded",
            "not_applicable_key": NOT_APPLICABLE_CLASS,
            "multi_label_field_weight_later": "mean of gold-label weights",
        },
        "invariants": invariants,
        "weight_map": weight_map,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--sft-split-manifest", default=DEFAULT_SFT_SPLIT_MANIFEST)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_path = _resolve(repo_root, args.output)
    artifact = build_cb_class_weight_artifact(
        repo_root=repo_root,
        data_path=args.data,
        manifest_path=args.manifest,
        split_manifest_path=args.sft_split_manifest,
        pack_path=args.pack,
    )
    write_exclusive_atomic_json(output_path, artifact)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "status": artifact["status"],
                "attribute_class_pairs": artifact["weight_map"][
                    "observed_attribute_class_pairs"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
