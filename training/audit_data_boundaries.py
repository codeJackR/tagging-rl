#!/usr/bin/env python3
"""Audit SKU and product-family boundaries across SFT, GRPO and eval data.

This command is deliberately read-only with respect to source datasets and
manifests. It verifies their hashes and membership, then writes one new,
collision-protected JSON report. It does not repair split assignments or build
a replacement GRPO pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from labeling.records import Row, read_jsonl
from training.split_sft import group_key

AUDIT_VERSION = "grpo-data-boundary-audit-v1"
FAMILY_GROUPING = (
    "training.split_sft.group_key: normalized brand + title before spaced "
    "variant separator"
)
DEFAULT_SFT_MANIFEST = "data/splits/sft-v1.json"
DEFAULT_POOL_MANIFEST = "runs/sft-difficulty-k8/grpo-pool-cap4-manifest.json"
DEFAULT_RUN1_ROLLOUTS = "../tagging-rl-artifacts/grpo-first-300/rollouts.jsonl"
DEFAULT_PROBE = "data/probe_100.jsonl"
DEFAULT_LEGACY_EVAL = "data/eval_300/eval.jsonl"
DEFAULT_OUTPUT = "runs/grpo-run2-data-boundary-audit.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sku_set_sha256(sku_ids: Sequence[str] | set[str]) -> str:
    payload = "\n".join(sorted(sku_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return value


def _read_generic_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _validate_unique_skus(name: str, sku_ids: Sequence[str]) -> list[str]:
    if any(not isinstance(sku, str) or not sku for sku in sku_ids):
        raise ValueError(f"{name} contains an invalid SKU ID")
    duplicates = sorted(
        sku for sku, count in Counter(sku_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{name} contains duplicate SKU IDs: {duplicates[:5]}")
    return list(sku_ids)


def _row_map(rows: Sequence[Row], *, source_name: str) -> dict[str, Row]:
    sku_ids = _validate_unique_skus(source_name, [row.sku_id for row in rows])
    return dict(zip(sku_ids, rows, strict=True))


def _merge_family_rows(*row_maps: Mapping[str, Row]) -> dict[str, Row]:
    merged: dict[str, Row] = {}
    for rows_by_sku in row_maps:
        for sku, row in rows_by_sku.items():
            previous = merged.get(sku)
            if previous is not None and group_key(previous) != group_key(row):
                raise ValueError(f"SKU {sku} has inconsistent product-family data")
            merged[sku] = row
    return merged


def _load_run1_skus(path: Path) -> tuple[list[str], dict]:
    records = _read_generic_jsonl(path)
    if not records:
        raise ValueError("Run 1 rollout file is empty")

    skus_by_step: dict[int, set[str]] = defaultdict(set)
    records_by_step: Counter[int] = Counter()
    first_step_by_sku: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        sku = record.get("sku_id")
        step = record.get("step")
        if not isinstance(sku, str) or not sku:
            raise ValueError(f"Run 1 rollout record {index} has invalid sku_id")
        if not isinstance(step, int) or step <= 0:
            raise ValueError(f"Run 1 rollout record {index} has invalid step")
        skus_by_step[step].add(sku)
        records_by_step[step] += 1
        first_step_by_sku.setdefault(sku, step)

    bad_steps = sorted(step for step, skus in skus_by_step.items() if len(skus) != 1)
    if bad_steps:
        raise ValueError(f"Run 1 rollout steps contain multiple SKUs: {bad_steps[:5]}")

    sku_step_counts = Counter(
        next(iter(skus)) for _, skus in sorted(skus_by_step.items())
    )
    repeated_skus = sorted(
        sku for sku, count in sku_step_counts.items() if count > 1
    )
    ordered_skus = [
        sku
        for sku, _ in sorted(first_step_by_sku.items(), key=lambda item: (item[1], item[0]))
    ]
    return ordered_skus, {
        "rollout_records": len(records),
        "steps": len(skus_by_step),
        "records_per_step_histogram": {
            str(count): frequency
            for count, frequency in sorted(Counter(records_by_step.values()).items())
        },
        "unique_skus": len(ordered_skus),
        "revisited_sku_count": len(repeated_skus),
        "revisited_skus": repeated_skus,
    }


def load_audit_inputs(
    *,
    repo_root: str | Path,
    sft_manifest_path: str | Path,
    pool_manifest_path: str | Path,
    run1_rollouts_path: str | Path,
    probe_path: str | Path,
    legacy_eval_path: str | Path,
) -> tuple[dict[str, list[str]], dict[str, Row], dict, dict]:
    """Load and hash-verify every source needed by the boundary audit."""
    repo_root = Path(repo_root).resolve()
    sft_manifest_file = _resolve(repo_root, sft_manifest_path)
    pool_manifest_file = _resolve(repo_root, pool_manifest_path)
    run1_rollouts_file = _resolve(repo_root, run1_rollouts_path)
    probe_file = _resolve(repo_root, probe_path)
    legacy_eval_file = _resolve(repo_root, legacy_eval_path)

    sft_manifest = _read_json(sft_manifest_file)
    sft_source_file = _resolve(repo_root, sft_manifest["source"])
    actual_sft_source_hash = sha256_file(sft_source_file)
    if actual_sft_source_hash != sft_manifest.get("source_sha256"):
        raise RuntimeError("SFT manifest source hash mismatch")

    sft_rows = read_jsonl(sft_source_file)
    sft_rows_by_sku = _row_map(sft_rows, source_name="SFT source")
    sft_train = _validate_unique_skus("SFT train manifest", sft_manifest["train"])
    sft_validation = _validate_unique_skus(
        "SFT validation manifest", sft_manifest["validation"]
    )
    train_set = set(sft_train)
    validation_set = set(sft_validation)
    if train_set & validation_set:
        raise ValueError("SFT manifest has train/validation SKU overlap")
    if train_set | validation_set != set(sft_rows_by_sku):
        raise ValueError("SFT manifest does not cover its source exactly")

    pool_manifest = _read_json(pool_manifest_file)
    pool_data_file = _resolve(repo_root, pool_manifest["output"]["active_dataset"])
    if sha256_file(pool_data_file) != pool_manifest["output"]["active_dataset_sha256"]:
        raise RuntimeError("GRPO pool manifest active-dataset hash mismatch")
    pool_rows = read_jsonl(pool_data_file)
    pool_rows_by_sku = _row_map(pool_rows, source_name="GRPO pool dataset")
    pool_skus = _validate_unique_skus(
        "GRPO pool manifest",
        pool_manifest["selection"]["active_skus_in_source_order"],
    )
    if len(pool_skus) != pool_manifest["output"]["active_dataset_rows"]:
        raise ValueError("GRPO pool manifest row count is inconsistent")
    if set(pool_skus) != set(pool_rows_by_sku):
        raise ValueError("GRPO pool manifest and active dataset SKU sets disagree")

    run1_skus, run1_structure = _load_run1_skus(run1_rollouts_file)
    probe_rows = read_jsonl(probe_file)
    probe_rows_by_sku = _row_map(probe_rows, source_name="probe dataset")
    legacy_eval_rows = read_jsonl(legacy_eval_file)
    legacy_eval_rows_by_sku = _row_map(
        legacy_eval_rows, source_name="legacy frozen eval dataset"
    )

    family_rows = _merge_family_rows(
        sft_rows_by_sku,
        pool_rows_by_sku,
        probe_rows_by_sku,
        legacy_eval_rows_by_sku,
    )
    collections = {
        "sft_train": sft_train,
        "sft_validation": sft_validation,
        "grpo_pool_cap4": pool_skus,
        "grpo_run1_trained": run1_skus,
        "probe_100": list(probe_rows_by_sku),
        "legacy_frozen_300": list(legacy_eval_rows_by_sku),
    }
    known_skus = set(family_rows)
    for name, sku_ids in collections.items():
        missing = sorted(set(sku_ids) - known_skus)
        if missing:
            raise ValueError(f"{name} has SKUs without family source rows: {missing[:5]}")

    inputs = {
        "sft_manifest": _file_metadata(sft_manifest_file),
        "sft_source": _file_metadata(sft_source_file),
        "grpo_pool_manifest": _file_metadata(pool_manifest_file),
        "grpo_pool_dataset": _file_metadata(pool_data_file),
        "grpo_run1_rollouts": _file_metadata(run1_rollouts_file),
        "probe_dataset": _file_metadata(probe_file),
        "legacy_frozen_dataset": _file_metadata(legacy_eval_file),
    }
    source_details = {
        "sft_manifest_version": sft_manifest.get("version"),
        "grpo_pool_manifest_version": pool_manifest.get("version"),
        "run1_rollout_structure": run1_structure,
        "embedded_split_counts": {
            "sft_source": dict(sorted(Counter(row.split for row in sft_rows).items())),
            "grpo_pool_dataset": dict(
                sorted(Counter(row.split for row in pool_rows).items())
            ),
            "probe_dataset": dict(
                sorted(Counter(row.split for row in probe_rows).items())
            ),
            "legacy_frozen_dataset": dict(
                sorted(Counter(row.split for row in legacy_eval_rows).items())
            ),
        },
    }
    return collections, family_rows, inputs, source_details


def _file_metadata(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _display_family(value: str) -> str:
    return value.replace("\0", "::")


def build_boundary_audit(
    *,
    collections: Mapping[str, Sequence[str]],
    family_rows: Mapping[str, Row],
    inputs: Mapping[str, dict],
    source_details: Mapping[str, object],
    code: Mapping[str, object],
) -> dict:
    """Build a deterministic overlap report from already validated inputs."""
    normalized: dict[str, list[str]] = {}
    for name in sorted(collections):
        normalized[name] = _validate_unique_skus(name, collections[name])

    all_skus = set().union(*(set(values) for values in normalized.values()))
    missing = sorted(all_skus - set(family_rows))
    if missing:
        raise ValueError(f"SKUs without product-family rows: {missing[:5]}")

    families_by_sku = {sku: group_key(family_rows[sku]) for sku in all_skus}
    sku_sets = {name: set(values) for name, values in normalized.items()}
    family_sets = {
        name: {families_by_sku[sku] for sku in values}
        for name, values in sku_sets.items()
    }
    authoritative_train = sku_sets["sft_train"]
    authoritative_validation = sku_sets["sft_validation"]

    summaries = {}
    for name in sorted(normalized):
        sku_ids = sku_sets[name]
        summaries[name] = {
            "rows": len(normalized[name]),
            "unique_skus": len(sku_ids),
            "sku_set_sha256": sku_set_sha256(sku_ids),
            "families": len(family_sets[name]),
            "authoritative_membership": {
                "sft_train": len(sku_ids & authoritative_train),
                "sft_validation": len(sku_ids & authoritative_validation),
                "outside_sft_source": len(
                    sku_ids - authoritative_train - authoritative_validation
                ),
            },
        }

    pairwise = []
    names = sorted(normalized)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            sku_overlap = sorted(sku_sets[left] & sku_sets[right])
            family_overlap = sorted(family_sets[left] & family_sets[right])
            family_overlap_set = set(family_overlap)
            left_family_member_skus = sorted(
                sku
                for sku in sku_sets[left]
                if families_by_sku[sku] in family_overlap_set
            )
            right_family_member_skus = sorted(
                sku
                for sku in sku_sets[right]
                if families_by_sku[sku] in family_overlap_set
            )
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "sku_overlap_count": len(sku_overlap),
                    "sku_overlap": sku_overlap,
                    "family_overlap_count": len(family_overlap),
                    "family_overlap": [
                        _display_family(value) for value in family_overlap
                    ],
                    "left_skus_in_overlapping_families_count": len(
                        left_family_member_skus
                    ),
                    "left_skus_in_overlapping_families": left_family_member_skus,
                    "right_skus_in_overlapping_families_count": len(
                        right_family_member_skus
                    ),
                    "right_skus_in_overlapping_families": right_family_member_skus,
                }
            )

    pool_validation_skus = sorted(
        sku_sets["grpo_pool_cap4"] & authoritative_validation
    )
    run1_validation_skus = sorted(
        sku_sets["grpo_run1_trained"] & authoritative_validation
    )
    pool_validation_families = sorted(
        family_sets["grpo_pool_cap4"] & family_sets["sft_validation"]
    )
    run1_validation_families = sorted(
        family_sets["grpo_run1_trained"] & family_sets["sft_validation"]
    )
    probe_pool_families = sorted(
        family_sets["probe_100"] & family_sets["grpo_pool_cap4"]
    )
    legacy_pool_families = sorted(
        family_sets["legacy_frozen_300"] & family_sets["grpo_pool_cap4"]
    )
    probe_run1_families = sorted(
        family_sets["probe_100"] & family_sets["grpo_run1_trained"]
    )
    legacy_run1_families = sorted(
        family_sets["legacy_frozen_300"] & family_sets["grpo_run1_trained"]
    )
    invariants = {
        "sft_train_validation_skus_disjoint": not (
            authoritative_train & authoritative_validation
        ),
        "sft_train_validation_families_disjoint": not (
            family_sets["sft_train"] & family_sets["sft_validation"]
        ),
        "grpo_pool_covered_by_sft_manifest": not (
            sku_sets["grpo_pool_cap4"]
            - authoritative_train
            - authoritative_validation
        ),
        "grpo_pool_has_zero_sft_validation_skus": not pool_validation_skus,
        "grpo_pool_has_zero_sft_validation_families": not pool_validation_families,
        "run1_products_are_subset_of_grpo_pool": (
            sku_sets["grpo_run1_trained"] <= sku_sets["grpo_pool_cap4"]
        ),
        "run1_has_zero_sft_validation_skus": not run1_validation_skus,
        "run1_has_zero_sft_validation_families": not run1_validation_families,
        "probe_has_zero_grpo_pool_skus": not (
            sku_sets["probe_100"] & sku_sets["grpo_pool_cap4"]
        ),
        "probe_has_zero_grpo_pool_families": not (
            family_sets["probe_100"] & family_sets["grpo_pool_cap4"]
        ),
        "legacy_frozen_has_zero_grpo_pool_skus": not (
            sku_sets["legacy_frozen_300"] & sku_sets["grpo_pool_cap4"]
        ),
        "legacy_frozen_has_zero_grpo_pool_families": not (
            family_sets["legacy_frozen_300"] & family_sets["grpo_pool_cap4"]
        ),
    }
    status = "passed" if all(invariants.values()) else "issues_found"

    return {
        "version": AUDIT_VERSION,
        "status": status,
        "authority": {
            "split_manifest": inputs["sft_manifest"]["path"],
            "split_manifest_sha256": inputs["sft_manifest"]["sha256"],
            "rule": (
                "SFT train/validation membership comes from the hash-verified "
                "external manifest; embedded row.split is diagnostic only"
            ),
            "family_grouping": FAMILY_GROUPING,
        },
        "code": dict(code),
        "inputs": {name: dict(inputs[name]) for name in sorted(inputs)},
        "source_details": dict(source_details),
        "collections": summaries,
        "pairwise_overlaps": pairwise,
        "headline_findings": {
            "grpo_pool_validation_sku_count": len(pool_validation_skus),
            "grpo_pool_validation_skus": pool_validation_skus,
            "grpo_pool_validation_family_count": len(pool_validation_families),
            "grpo_pool_validation_families": [
                _display_family(value) for value in pool_validation_families
            ],
            "run1_validation_sku_count": len(run1_validation_skus),
            "run1_validation_skus": run1_validation_skus,
            "run1_validation_family_count": len(run1_validation_families),
            "run1_validation_families": [
                _display_family(value) for value in run1_validation_families
            ],
            "probe_grpo_pool_family_count": len(probe_pool_families),
            "probe_grpo_pool_families": [
                _display_family(value) for value in probe_pool_families
            ],
            "legacy_frozen_grpo_pool_family_count": len(legacy_pool_families),
            "legacy_frozen_grpo_pool_families": [
                _display_family(value) for value in legacy_pool_families
            ],
            "probe_run1_family_count": len(probe_run1_families),
            "probe_run1_families": [
                _display_family(value) for value in probe_run1_families
            ],
            "legacy_frozen_run1_family_count": len(legacy_run1_families),
            "legacy_frozen_run1_families": [
                _display_family(value) for value in legacy_run1_families
            ],
        },
        "invariants": invariants,
    }


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("git rev-parse returned an invalid commit")
    return value


def write_exclusive_atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--sft-manifest", default=DEFAULT_SFT_MANIFEST)
    parser.add_argument("--pool-manifest", default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--run1-rollouts", default=DEFAULT_RUN1_ROLLOUTS)
    parser.add_argument("--probe", default=DEFAULT_PROBE)
    parser.add_argument("--legacy-eval", default=DEFAULT_LEGACY_EVAL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    collections, family_rows, inputs, source_details = load_audit_inputs(
        repo_root=repo_root,
        sft_manifest_path=args.sft_manifest,
        pool_manifest_path=args.pool_manifest,
        run1_rollouts_path=args.run1_rollouts,
        probe_path=args.probe,
        legacy_eval_path=args.legacy_eval,
    )
    implementation = Path(__file__).resolve()
    audit = build_boundary_audit(
        collections=collections,
        family_rows=family_rows,
        inputs=inputs,
        source_details=source_details,
        code={
            "git_commit": _git_commit(repo_root),
            "implementation_file": str(implementation.relative_to(repo_root)),
            "implementation_file_sha256": sha256_file(implementation),
        },
    )
    output = _resolve(repo_root, args.output)
    write_exclusive_atomic_json(output, audit)
    findings = audit["headline_findings"]
    print(
        f"boundary audit {audit['status']}: "
        f"pool_validation={findings['grpo_pool_validation_sku_count']}, "
        f"run1_validation={findings['run1_validation_sku_count']}, "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
