#!/usr/bin/env python3
"""Build the fixed, grouped train/validation split used by every SFT arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from labeling.records import read_jsonl

SPLIT_VERSION = "sft-v1"
_VARIANT_SEPARATOR = re.compile(r"\s+(?:\||/|[-–—])\s+")
_TRAILING_PARENS = re.compile(r"\s*\([^)]*\)\s*$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(value: str) -> str:
    return _NON_ALNUM.sub(" ", value.casefold()).strip()


def family_title(title: str) -> str:
    """Remove conservative retail variant suffixes, then normalize.

    Examples:
      "Everyday Tote | Red (XL)" -> "everyday tote"
      "Classic Tee - Navy"       -> "classic tee"

    We only compare within the same brand. Broad fuzzy title matching would risk
    merging unrelated products such as two different brands' "Classic Tee".
    """
    base = _VARIANT_SEPARATOR.split(title, maxsplit=1)[0]
    base = _TRAILING_PARENS.sub("", base)
    return normalize(base)


def group_key(row) -> str:
    brand = normalize(row.input.brand or "<missing-brand>")
    return f"{brand}\0{family_title(row.input.title)}"


def stable_order(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest()


def row_category(row) -> str:
    labels = getattr(row, "labels", {})
    label = labels.get("garment_category") if labels else None
    if label is None:
        return "<unknown>"
    return str(label.value or label.status.value)


def category_targets(rows, validation_size: int) -> dict[str, int]:
    """Largest-remainder allocation of the validation budget by category."""
    counts = Counter(row_category(row) for row in rows)
    exact = {
        category: count * validation_size / len(rows)
        for category, count in counts.items()
    }
    targets = {category: math.floor(value) for category, value in exact.items()}
    remaining = validation_size - sum(targets.values())
    order = sorted(
        counts,
        key=lambda category: (
            -(exact[category] - targets[category]),
            category,
        ),
    )
    for category in order[:remaining]:
        targets[category] += 1
    return targets


def build_assignments(rows, *, validation_size: int, seed: int) -> dict:
    if not 0 < validation_size < len(rows):
        raise ValueError("validation_size must be between 1 and len(rows)-1")

    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    targets = category_targets(rows, validation_size)
    groups_by_category: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        categories = Counter(row_category(row) for row in members)
        dominant = sorted(categories, key=lambda value: (-categories[value], value))[0]
        groups_by_category[dominant].append(key)

    validation: set[str] = set()
    selected_groups: set[str] = set()
    for category, target in sorted(targets.items()):
        remaining_for_category = target
        keys = sorted(
            groups_by_category[category],
            key=lambda value: stable_order(value, seed),
        )
        for key in keys:
            group = groups[key]
            if len(group) <= remaining_for_category:
                selected_groups.add(key)
                validation.update(row.sku_id for row in group)
                remaining_for_category -= len(group)
            if remaining_for_category == 0:
                break

    # Mixed-category families or unusually large families can leave a quota a few
    # rows short. Fill any remainder with whole, still-unselected groups.
    remaining = validation_size - len(validation)
    if remaining:
        for key in sorted(groups, key=lambda value: stable_order(value, seed)):
            if key in selected_groups:
                continue
            group = groups[key]
            if len(group) <= remaining:
                selected_groups.add(key)
                validation.update(row.sku_id for row in group)
                remaining -= len(group)
            if remaining == 0:
                break

    if remaining:
        raise RuntimeError(
            f"could not fill validation target; {remaining} rows remain "
            "(grouping unexpectedly has no small groups)"
        )

    train = [row.sku_id for row in rows if row.sku_id not in validation]
    val = [row.sku_id for row in rows if row.sku_id in validation]
    grouped = [members for members in groups.values() if len(members) > 1]
    return {
        "version": SPLIT_VERSION,
        "seed": seed,
        "grouping": (
            "normalized brand + title before spaced variant separator; "
            "group-stratified by garment_category"
        ),
        "n_rows": len(rows),
        "n_groups": len(groups),
        "n_variant_groups": len(grouped),
        "n_rows_in_variant_groups": sum(len(group) for group in grouped),
        "validation_category_targets": targets,
        "validation_category_counts": dict(
            sorted(Counter(row_category(row) for row in rows if row.sku_id in validation).items())
        ),
        "train": train,
        "validation": val,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/train_weak.jsonl")
    parser.add_argument("--output", default="data/splits/sft-v1.json")
    parser.add_argument("--validation-size", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists; pass --overwrite to replace it")

    rows = read_jsonl(source)
    manifest = build_assignments(
        rows,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    manifest["source"] = str(source)
    manifest["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"{manifest['version']}: {len(manifest['train'])} train, "
        f"{len(manifest['validation'])} validation, "
        f"{manifest['n_variant_groups']} variant groups"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
