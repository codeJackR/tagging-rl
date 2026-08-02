#!/usr/bin/env python3
"""Build a provenance-rich composition audit for the retained GRPO pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from labeling.records import LabelStatus, read_jsonl
from training.split_sft import family_title, group_key, row_category

AUDIT_VERSION = "grpo-retained-pool-audit-v1"
DEFAULT_SCORED_DATA = "data/train_weak_sft_scored.jsonl"
DEFAULT_DIFFICULTY_MANIFEST = "runs/sft-difficulty-k8/manifest.json"
DEFAULT_OUTPUT = "runs/sft-difficulty-k8/retained-pool-audit.json"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _store(row) -> str:
    parts = row.sku_id.split(":", 2)
    return parts[1] if len(parts) == 3 else "<unknown>"


def _composition(rows, retained, key_fn: Callable) -> dict:
    full_counts = Counter(key_fn(row) for row in rows)
    retained_counts = Counter(key_fn(row) for row in retained)
    groups = []
    for key in sorted(full_counts):
        full_count = full_counts[key]
        retained_count = retained_counts[key]
        full_share = full_count / len(rows)
        retained_share = retained_count / len(retained)
        groups.append(
            {
                "key": key,
                "full_rows": full_count,
                "retained_rows": retained_count,
                "retention_rate": retained_count / full_count,
                "full_share": full_share,
                "retained_share": retained_share,
                "share_shift_percentage_points": 100
                * (retained_share - full_share),
            }
        )
    return {
        "full_group_count": len(full_counts),
        "retained_group_count": sum(count > 0 for count in retained_counts.values()),
        "total_variation_distance": 0.5
        * sum(abs(group["retained_share"] - group["full_share"]) for group in groups),
        "groups": groups,
    }


def _gold_density(rows) -> dict:
    scorable = []
    substantive = []
    not_applicable = []
    unknown = []
    for row in rows:
        counts = Counter(label.status for label in row.labels.values())
        labeled = counts[LabelStatus.LABELED]
        na = counts[LabelStatus.NOT_APPLICABLE]
        unk = counts[LabelStatus.UNKNOWN]
        substantive.append(labeled)
        not_applicable.append(na)
        unknown.append(unk)
        scorable.append(labeled + na)
    return {
        "rows": len(rows),
        "mean_scorable": sum(scorable) / len(rows),
        "mean_substantive_labeled": sum(substantive) / len(rows),
        "mean_not_applicable": sum(not_applicable) / len(rows),
        "mean_unknown": sum(unknown) / len(rows),
        "minimum_scorable": min(scorable),
        "maximum_scorable": max(scorable),
        "rows_with_at_most_2_scorable": sum(value <= 2 for value in scorable),
        "rows_with_at_most_2_substantive": sum(
            value <= 2 for value in substantive
        ),
    }


def _family_audit(rows, retained) -> dict:
    retained_ids = {row.sku_id for row in retained}
    families = defaultdict(list)
    for row in rows:
        families[group_key(row)].append(row)

    represented = []
    for key, members in families.items():
        kept = [row for row in members if row.sku_id in retained_ids]
        if not kept:
            continue
        represented.append(
            {
                "key": key.replace("\0", "::"),
                "brand": members[0].input.brand,
                "family_title": family_title(members[0].input.title),
                "full_rows": len(members),
                "retained_rows": len(kept),
                "retained_share": len(kept) / len(retained),
                "pass_rate_histogram": dict(
                    sorted(
                        Counter(
                            f"{row.difficulty.sft_pass_rate:.3f}" for row in members
                        ).items()
                    )
                ),
            }
        )

    represented.sort(
        key=lambda item: (
            -item["retained_rows"],
            -item["full_rows"],
            item["key"],
        )
    )
    retained_histogram = Counter(item["retained_rows"] for item in represented)
    variant_represented = [item for item in represented if item["full_rows"] > 1]
    return {
        "grouping": (
            "training.split_sft.group_key: normalized brand + title before "
            "spaced variant separator"
        ),
        "full_families": len(families),
        "full_variant_families": sum(
            len(members) > 1 for members in families.values()
        ),
        "full_rows_in_variant_families": sum(
            len(members) for members in families.values() if len(members) > 1
        ),
        "retained_families": len(represented),
        "retained_rows_per_family": len(retained) / len(represented),
        "retained_duplicate_rows_beyond_one_per_family": (
            len(retained) - len(represented)
        ),
        "represented_variant_families": len(variant_represented),
        "retained_rows_from_variant_families": sum(
            item["retained_rows"] for item in variant_represented
        ),
        "fully_retained_variant_families": sum(
            item["retained_rows"] == item["full_rows"]
            for item in variant_represented
        ),
        "partially_retained_variant_families": sum(
            item["retained_rows"] < item["full_rows"]
            for item in variant_represented
        ),
        "retained_family_size_histogram": {
            str(size): count for size, count in sorted(retained_histogram.items())
        },
        "maximum_retained_rows_one_family": represented[0]["retained_rows"],
        "top_20_retained_families": represented[:20],
    }


def _sku_set_sha256(rows) -> str:
    payload = "\n".join(sorted(row.sku_id for row in rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _weighted_distribution(rows_with_probability, key_fn: Callable) -> Counter:
    distribution = Counter()
    for row, probability in rows_with_probability:
        distribution[key_fn(row)] += probability
    return distribution


def _total_variation(first: Counter, second: Counter) -> float:
    keys = set(first) | set(second)
    return 0.5 * sum(abs(first[key] - second[key]) for key in keys)


def select_deterministic_family_cap(rows, *, cap: int, seed: int):
    """Keep at most ``cap`` rows per canonical family, preserving source order."""
    if cap <= 0:
        raise ValueError("family cap must be positive")
    families = defaultdict(list)
    for row in rows:
        families[group_key(row)].append(row)

    selected_ids = set()
    for key in sorted(families):
        members = sorted(
            families[key],
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row.sku_id}".encode("utf-8")
            ).hexdigest(),
        )
        selected_ids.update(row.sku_id for row in members[:cap])
    return [row for row in rows if row.sku_id in selected_ids]


def _sampling_policy_comparison(rows, retained, *, cap_seed: int = 42) -> dict:
    families = defaultdict(list)
    for row in retained:
        families[group_key(row)].append(row)

    full_category = Counter(row_category(row) for row in rows)
    full_category = Counter(
        {key: count / len(rows) for key, count in full_category.items()}
    )
    full_store = Counter(_store(row) for row in rows)
    full_store = Counter(
        {key: count / len(rows) for key, count in full_store.items()}
    )
    retained_difficulty = Counter(
        f"{row.difficulty.sft_pass_rate:.3f}" for row in retained
    )
    retained_difficulty = Counter(
        {
            key: count / len(retained)
            for key, count in retained_difficulty.items()
        }
    )

    def scenario(name: str, active_rows, weighted_rows) -> dict:
        family_probability = _weighted_distribution(
            weighted_rows, lambda row: group_key(row)
        )
        category_probability = _weighted_distribution(
            weighted_rows, row_category
        )
        store_probability = _weighted_distribution(weighted_rows, _store)
        difficulty_probability = _weighted_distribution(
            weighted_rows, lambda row: f"{row.difficulty.sft_pass_rate:.3f}"
        )
        probabilities = sorted(family_probability.values(), reverse=True)
        return {
            "policy": name,
            "active_rows": len(active_rows),
            "active_row_share": len(active_rows) / len(retained),
            "active_sku_set_sha256": _sku_set_sha256(active_rows),
            "families": len(family_probability),
            "maximum_family_probability": probabilities[0],
            "top_5_family_probability": sum(probabilities[:5]),
            "top_10_family_probability": sum(probabilities[:10]),
            "effective_family_count_inverse_hhi": 1
            / sum(probability**2 for probability in probabilities),
            "category_tvd_vs_full": _total_variation(
                category_probability, full_category
            ),
            "store_tvd_vs_full": _total_variation(
                store_probability, full_store
            ),
            "difficulty_tvd_vs_eligible": _total_variation(
                difficulty_probability, retained_difficulty
            ),
            "expected_pass_rate": sum(
                row.difficulty.sft_pass_rate * probability
                for row, probability in weighted_rows
            ),
        }

    scenarios = []
    row_probability = 1 / len(retained)
    scenarios.append(
        scenario(
            "row_uniform",
            retained,
            [(row, row_probability) for row in retained],
        )
    )

    for cap in (2, 4, 8):
        selected = select_deterministic_family_cap(
            retained, cap=cap, seed=cap_seed
        )
        probability = 1 / len(selected)
        scenarios.append(
            scenario(
                f"deterministic_family_cap_{cap}",
                selected,
                [(row, probability) for row in selected],
            )
        )

    family_uniform_rows = []
    family_probability = 1 / len(families)
    for members in families.values():
        row_probability = family_probability / len(members)
        family_uniform_rows.extend(
            (row, row_probability) for row in members
        )
    scenarios.append(
        scenario("family_uniform", retained, family_uniform_rows)
    )
    return {
        "cap_selection": (
            "within each canonical family, sort SKU IDs by SHA-256 of "
            f"'{cap_seed}\\0<sku_id>' and keep the first cap rows"
        ),
        "cap_seed": cap_seed,
        "family_uniform_definition": (
            "choose one canonical family uniformly, then one retained row "
            "uniformly within that family"
        ),
        "scenarios": scenarios,
    }


def build_audit(
    rows,
    *,
    scored_path: str | Path,
    difficulty_manifest_path: str | Path,
    created_at_utc: str,
) -> dict:
    retained = [
        row
        for row in rows
        if row.difficulty.sft_pass_rate is not None
        and 0 < row.difficulty.sft_pass_rate < 1
    ]
    if not retained:
        raise ValueError("retained GRPO pool is empty")
    if len({row.sku_id for row in rows}) != len(rows):
        raise ValueError("scored dataset contains duplicate SKU IDs")

    difficulty_manifest_path = Path(difficulty_manifest_path)
    difficulty_manifest = json.loads(
        difficulty_manifest_path.read_text(encoding="utf-8")
    )
    expected_rows = difficulty_manifest["summary"]["products_scored"]
    expected_retained = difficulty_manifest["summary"]["retained_for_grpo"]
    if len(rows) != expected_rows or len(retained) != expected_retained:
        raise ValueError("scored data disagrees with difficulty manifest counts")

    categories = _composition(rows, retained, row_category)
    stores = _composition(rows, retained, _store)
    sources = _composition(rows, retained, lambda row: row.source)
    brands = _composition(
        rows, retained, lambda row: row.input.brand or "<missing-brand>"
    )
    families = _family_audit(rows, retained)

    def largest_shift(composition: dict, direction: int) -> dict:
        supported = [
            group for group in composition["groups"] if group["full_rows"] >= 10
        ]
        return sorted(
            supported,
            key=lambda group: (
                direction * group["share_shift_percentage_points"],
                group["key"],
            ),
        )[0]

    return {
        "version": AUDIT_VERSION,
        "created_at_utc": created_at_utc,
        "inputs": {
            "scored_dataset": str(scored_path),
            "scored_dataset_sha256": _sha256_file(scored_path),
            "difficulty_manifest": str(difficulty_manifest_path),
            "difficulty_manifest_sha256": _sha256_file(difficulty_manifest_path),
            "difficulty_manifest_version": difficulty_manifest["version"],
        },
        "selection": {
            "rule": "0 < sft_pass_rate < 1",
            "full_rows": len(rows),
            "retained_rows": len(retained),
            "retained_share": len(retained) / len(rows),
            "retained_sku_set_sha256": _sku_set_sha256(retained),
        },
        "composition": {
            "sources": sources,
            "stores": stores,
            "brands": brands,
            "garment_categories": categories,
        },
        "families": families,
        "gold_density": {
            "full": _gold_density(rows),
            "retained": _gold_density(retained),
        },
        "sampling_policy_comparison": _sampling_policy_comparison(rows, retained),
        "headline_findings": {
            "all_stores_represented": (
                stores["full_group_count"] == stores["retained_group_count"]
            ),
            "missing_categories": [
                group["key"]
                for group in categories["groups"]
                if group["retained_rows"] == 0
            ],
            "largest_supported_category_increase": largest_shift(categories, -1),
            "largest_supported_category_decrease": largest_shift(categories, 1),
            "largest_supported_store_increase": largest_shift(stores, -1),
            "largest_supported_store_decrease": largest_shift(stores, 1),
            "largest_family": families["top_20_retained_families"][0],
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-data", default=DEFAULT_SCORED_DATA)
    parser.add_argument(
        "--difficulty-manifest", default=DEFAULT_DIFFICULTY_MANIFEST
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    rows = read_jsonl(args.scored_data)
    audit = build_audit(
        rows,
        scored_path=args.scored_data,
        difficulty_manifest_path=args.difficulty_manifest,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"retained-pool audit complete: {audit['selection']['retained_rows']} rows, "
        f"{audit['families']['retained_families']} families, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
