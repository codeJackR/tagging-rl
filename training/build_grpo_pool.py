#!/usr/bin/env python3
"""Build the locked cap-four active GRPO dataset and selection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from labeling.records import LabelStatus, read_jsonl, write_jsonl
from training.audit_grpo_pool import (
    AUDIT_VERSION,
    _sampling_policy_comparison,
    select_deterministic_family_cap,
)
from training.split_sft import group_key, row_category

ROOT = Path(__file__).resolve().parent.parent
POOL_VERSION = "grpo-pool-cap4-v1"
DEFAULT_SCORED_DATA = "data/train_weak_sft_scored.jsonl"
DEFAULT_AUDIT = "runs/sft-difficulty-k8/retained-pool-audit.json"
DEFAULT_OUTPUT_DATA = "data/train_weak_grpo_cap4.jsonl"
DEFAULT_OUTPUT_MANIFEST = "runs/sft-difficulty-k8/grpo-pool-cap4-manifest.json"
DEFAULT_CAP = 4
DEFAULT_SEED = 42


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_authoritative_training_rows(
    rows,
    *,
    split_manifest_path: str | Path,
):
    """Select SFT-training rows using the external manifest, not ``row.split``.

    The embedded split marks every member of the 3,600-row weak-training corpus
    as ``train``. The later SFT manifest subdivides that corpus into 3,240 SFT
    training and 360 validation rows, so it is the authoritative boundary for
    GRPO eligibility.
    """
    split_manifest_path = Path(split_manifest_path)
    if not split_manifest_path.is_absolute():
        split_manifest_path = ROOT / split_manifest_path
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    source_path = Path(manifest["source"])
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if _sha256_file(source_path) != manifest.get("source_sha256"):
        raise ValueError("SFT split manifest source hash mismatch")

    train_ids = manifest.get("train")
    validation_ids = manifest.get("validation")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise ValueError("SFT split manifest assignments must be lists")
    if any(not isinstance(sku, str) or not sku for sku in train_ids + validation_ids):
        raise ValueError("SFT split manifest contains an invalid SKU ID")
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("SFT split manifest contains duplicate training SKUs")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("SFT split manifest contains duplicate validation SKUs")

    train_set = set(train_ids)
    validation_set = set(validation_ids)
    if train_set & validation_set:
        raise ValueError("SFT split manifest has train/validation overlap")

    source_rows = read_jsonl(source_path)
    source_ids = [row.sku_id for row in source_rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("SFT split source contains duplicate SKU IDs")
    if train_set | validation_set != set(source_ids):
        raise ValueError("SFT split manifest does not cover its source exactly")

    source_by_sku = {row.sku_id: row for row in source_rows}
    train_families = {group_key(source_by_sku[sku]) for sku in train_set}
    validation_families = {
        group_key(source_by_sku[sku]) for sku in validation_set
    }
    overlapping_families = sorted(train_families & validation_families)
    if overlapping_families:
        examples = ", ".join(
            family.replace("\0", "::") for family in overlapping_families[:5]
        )
        raise ValueError(
            "SFT split manifest has family overlap: "
            f"{len(overlapping_families)} families ({examples})"
        )

    rows = list(rows)
    row_ids = [row.sku_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("candidate dataset contains duplicate SKU IDs")
    if set(row_ids) != set(source_ids):
        raise ValueError("candidate dataset does not match the SFT split source")

    selected = [row for row in rows if row.sku_id in train_set]
    if {row.sku_id for row in selected} != train_set:
        raise RuntimeError("authoritative SFT training filter produced wrong SKU set")
    return selected


def _sku_set_sha256(rows) -> str:
    payload = "\n".join(sorted(row.sku_id for row in rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _store(row) -> str:
    parts = row.sku_id.split(":", 2)
    return parts[1] if len(parts) == 3 else "<unknown>"


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], check=False
        ).returncode != 0
        return {"git_commit": commit, "tracked_worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "tracked_worktree_dirty": None}


def _gold_density(rows) -> dict:
    scorable = []
    substantive = []
    for row in rows:
        counts = Counter(label.status for label in row.labels.values())
        scorable.append(
            counts[LabelStatus.LABELED] + counts[LabelStatus.NOT_APPLICABLE]
        )
        substantive.append(counts[LabelStatus.LABELED])
    return {
        "mean_scorable": sum(scorable) / len(rows),
        "mean_substantive_labeled": sum(substantive) / len(rows),
        "minimum_scorable": min(scorable),
        "maximum_scorable": max(scorable),
        "rows_with_at_most_2_scorable": sum(value <= 2 for value in scorable),
    }


def select_pool(rows, audit: dict, *, cap: int, seed: int):
    if audit.get("version") != AUDIT_VERSION:
        raise ValueError("unexpected retained-pool audit version")
    eligible = [
        row
        for row in rows
        if row.difficulty.sft_pass_rate is not None
        and 0 < row.difficulty.sft_pass_rate < 1
    ]
    if len({row.sku_id for row in rows}) != len(rows):
        raise ValueError("scored dataset contains duplicate SKU IDs")
    if len(eligible) != audit["selection"]["retained_rows"]:
        raise ValueError("eligible-row count disagrees with audit")
    if _sku_set_sha256(eligible) != audit["selection"]["retained_sku_set_sha256"]:
        raise ValueError("eligible SKU set disagrees with audit")

    policy_name = f"deterministic_family_cap_{cap}"
    scenarios = {
        scenario["policy"]: scenario
        for scenario in audit["sampling_policy_comparison"]["scenarios"]
    }
    if policy_name not in scenarios:
        raise ValueError(f"audit does not contain policy {policy_name}")
    if seed != audit["sampling_policy_comparison"]["cap_seed"]:
        raise ValueError("cap seed disagrees with audit")

    active = select_deterministic_family_cap(eligible, cap=cap, seed=seed)
    active_ids = {row.sku_id for row in active}
    capped = [row for row in eligible if row.sku_id not in active_ids]
    scenario = scenarios[policy_name]
    if len(active) != scenario["active_rows"]:
        raise ValueError("active-row count disagrees with audit")
    if _sku_set_sha256(active) != scenario["active_sku_set_sha256"]:
        raise ValueError("active SKU set disagrees with audit")
    return eligible, active, capped, scenario


def _active_composition(active) -> dict:
    families = defaultdict(list)
    for row in active:
        families[group_key(row)].append(row)
    return {
        "garment_category_counts": dict(
            sorted(Counter(row_category(row) for row in active).items())
        ),
        "store_counts": dict(sorted(Counter(_store(row) for row in active).items())),
        "source_counts": dict(
            sorted(Counter(row.source for row in active).items())
        ),
        "pass_rate_histogram": dict(
            sorted(
                Counter(
                    f"{row.difficulty.sft_pass_rate:.3f}" for row in active
                ).items()
            )
        ),
        "family_count": len(families),
        "maximum_rows_one_family": max(len(members) for members in families.values()),
        "family_size_histogram": {
            str(size): count
            for size, count in sorted(
                Counter(len(members) for members in families.values()).items()
            )
        },
        "gold_density": _gold_density(active),
    }


def build_pool_manifest(
    *,
    rows,
    active,
    capped,
    scenario: dict,
    scored_path: str | Path,
    audit_path: str | Path,
    output_data_path: str | Path,
    cap: int,
    seed: int,
    created_at_utc: str,
    split_manifest_path: str | Path | None = None,
) -> dict:
    output_rows = read_jsonl(output_data_path)
    if [row.sku_id for row in output_rows] != [row.sku_id for row in active]:
        raise ValueError("written active dataset order or membership changed")
    eligible = [
        row
        for row in rows
        if row.difficulty.sft_pass_rate is not None
        and 0 < row.difficulty.sft_pass_rate < 1
    ]
    eligible_ids = {row.sku_id for row in eligible}
    active_ids = {row.sku_id for row in active}
    capped_ids = {row.sku_id for row in capped}
    if active_ids & capped_ids or active_ids | capped_ids != eligible_ids:
        raise ValueError("active and capped SKU sets do not partition eligibility")

    implementation_path = Path(__file__)
    try:
        implementation_display = str(implementation_path.resolve().relative_to(Path.cwd()))
    except ValueError:
        implementation_display = str(implementation_path)
    manifest_inputs = {
        "scored_dataset": str(scored_path),
        "scored_dataset_sha256": _sha256_file(scored_path),
        "retained_pool_audit": str(audit_path),
        "retained_pool_audit_sha256": _sha256_file(audit_path),
        "retained_pool_audit_version": AUDIT_VERSION,
    }
    authoritative_train_ids = None
    authoritative_validation_ids = None
    authoritative_validation_families = None
    if split_manifest_path is not None:
        resolved_split_manifest = Path(split_manifest_path)
        if not resolved_split_manifest.is_absolute():
            resolved_split_manifest = ROOT / resolved_split_manifest
        split_manifest = json.loads(
            resolved_split_manifest.read_text(encoding="utf-8")
        )
        authoritative_train_ids = set(split_manifest["train"])
        authoritative_validation_ids = set(split_manifest["validation"])
        split_source = Path(split_manifest["source"])
        if not split_source.is_absolute():
            split_source = ROOT / split_source
        split_rows_by_sku = {
            row.sku_id: row for row in read_jsonl(split_source)
        }
        authoritative_validation_families = {
            group_key(split_rows_by_sku[sku])
            for sku in authoritative_validation_ids
        }
        manifest_inputs.update(
            {
                "sft_split_manifest": str(split_manifest_path),
                "sft_split_manifest_sha256": _sha256_file(
                    resolved_split_manifest
                ),
                "sft_split_manifest_version": split_manifest.get("version"),
            }
        )

    invariants = {
        "all_active_rows_are_eligible": all(
            0 < row.difficulty.sft_pass_rate < 1 for row in active
        ),
        "active_and_capped_are_disjoint": not bool(active_ids & capped_ids),
        "active_and_capped_cover_eligible": (
            active_ids | capped_ids == eligible_ids
        ),
        "all_eligible_families_represented": (
            {group_key(row) for row in active}
            == {group_key(row) for row in eligible}
        ),
        "family_cap_respected": (
            max(Counter(group_key(row) for row in active).values()) <= cap
        ),
    }
    if authoritative_train_ids is not None:
        invariants["all_active_rows_are_authoritative_sft_train"] = (
            active_ids <= authoritative_train_ids
        )
        invariants["active_and_sft_validation_skus_are_disjoint"] = not (
            active_ids & authoritative_validation_ids
        )
        invariants["active_and_sft_validation_families_are_disjoint"] = not (
            {group_key(row) for row in active}
            & authoritative_validation_families
        )

    return {
        "version": (
            "grpo-pool-cap4-sft-train-v1"
            if split_manifest_path is not None
            else POOL_VERSION
        ),
        "created_at_utc": created_at_utc,
        "inputs": manifest_inputs,
        "policy": {
            "eligibility_rule": "0 < sft_pass_rate < 1",
            "family_grouping": (
                "training.split_sft.group_key: normalized brand + title before "
                "spaced variant separator"
            ),
            "family_cap": cap,
            "selection_seed": seed,
            "within_family_order": (
                f"SHA-256 of '{seed}\\0<sku_id>', ascending"
            ),
            "audited_scenario": scenario,
        },
        "selection": {
            "source_rows": len(rows),
            "eligible_rows": len(eligible),
            "active_rows": len(active),
            "capped_eligible_rows": len(capped),
            "eligible_sku_set_sha256": _sku_set_sha256(eligible),
            "active_sku_set_sha256": _sku_set_sha256(active),
            "capped_sku_set_sha256": _sku_set_sha256(capped),
            "active_skus_in_source_order": [row.sku_id for row in active],
            "capped_eligible_skus_in_source_order": [
                row.sku_id for row in capped
            ],
        },
        "output": {
            "active_dataset": str(output_data_path),
            "active_dataset_rows": len(output_rows),
            "active_dataset_bytes": Path(output_data_path).stat().st_size,
            "active_dataset_sha256": _sha256_file(output_data_path),
        },
        "active_composition": _active_composition(active),
        "invariants": invariants,
        "code": {
            **_git_state(),
            "implementation_file": implementation_display,
            "implementation_file_sha256": _sha256_file(implementation_path),
        },
    }


def write_pool_artifacts(
    *,
    scored_path: str | Path,
    audit_path: str | Path,
    output_data_path: str | Path,
    output_manifest_path: str | Path,
    cap: int,
    seed: int,
    created_at_utc: str,
    overwrite: bool = False,
    split_manifest_path: str | Path | None = None,
) -> dict:
    output_data_path = Path(output_data_path)
    output_manifest_path = Path(output_manifest_path)
    collisions = [
        path for path in (output_data_path, output_manifest_path) if path.exists()
    ]
    if collisions and not overwrite:
        raise FileExistsError(
            "output already exists: " + ", ".join(str(path) for path in collisions)
        )

    rows = read_jsonl(scored_path)
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    # Validate the historical all-3,600-row difficulty audit before applying a
    # narrower authoritative training boundary. This preserves its provenance
    # while preventing its embedded W1-level split from controlling GRPO.
    eligible, active, capped, scenario = select_pool(
        rows, audit, cap=cap, seed=seed
    )
    if split_manifest_path is not None:
        rows = filter_authoritative_training_rows(
            rows,
            split_manifest_path=split_manifest_path,
        )
        eligible = [
            row
            for row in rows
            if row.difficulty.sft_pass_rate is not None
            and 0 < row.difficulty.sft_pass_rate < 1
        ]
        comparison = _sampling_policy_comparison(
            rows,
            eligible,
            cap_seed=seed,
        )
        policy_name = f"deterministic_family_cap_{cap}"
        scenario = next(
            item
            for item in comparison["scenarios"]
            if item["policy"] == policy_name
        )
        active = select_deterministic_family_cap(
            eligible,
            cap=cap,
            seed=seed,
        )
        active_ids = {row.sku_id for row in active}
        capped = [row for row in eligible if row.sku_id not in active_ids]
        if _sku_set_sha256(active) != scenario["active_sku_set_sha256"]:
            raise RuntimeError("authoritative pool disagrees with recomputed scenario")
    if len(eligible) != len(active) + len(capped):
        raise ValueError("eligible partition count mismatch")
    write_jsonl(active, output_data_path)
    manifest = build_pool_manifest(
        rows=rows,
        active=active,
        capped=capped,
        scenario=scenario,
        scored_path=scored_path,
        audit_path=audit_path,
        output_data_path=output_data_path,
        cap=cap,
        seed=seed,
        created_at_utc=created_at_utc,
        split_manifest_path=split_manifest_path,
    )
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-data", default=DEFAULT_SCORED_DATA)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument(
        "--sft-split-manifest",
        default=None,
        help=(
            "authoritative SFT train/validation manifest; when supplied, only "
            "manifest-training rows may enter the new GRPO pool"
        ),
    )
    parser.add_argument("--output-data", default=DEFAULT_OUTPUT_DATA)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--family-cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = write_pool_artifacts(
        scored_path=args.scored_data,
        audit_path=args.audit,
        output_data_path=args.output_data,
        output_manifest_path=args.output_manifest,
        cap=args.family_cap,
        seed=args.seed,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        overwrite=args.overwrite,
        split_manifest_path=args.sft_split_manifest,
    )
    print(
        "GRPO pool complete: "
        f"eligible={manifest['selection']['eligible_rows']}, "
        f"active={manifest['selection']['active_rows']}, "
        f"capped={manifest['selection']['capped_eligible_rows']}, "
        f"manifest={args.output_manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
