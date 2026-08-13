"""Build the GRPO Run 2 development-role manifest without model inference.

The authoritative SFT validation split becomes one disclosed development
dataset with pre-outcome difficulty views.  This builder deliberately leaves
final confirmation unassigned; it must not silently promote an already-used
dataset into an untouched role.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from labeling.records import read_jsonl
from training.split_sft import group_key


VERSION = "grpo-run2-data-role-manifest-v1"
DEFAULT_OUTPUT = "runs/grpo-run2-data-role-manifest.json"
MINIMUM_INTERPRETABLE_ROWS = 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def build_development_views(
    *,
    validation_order: Sequence[str],
    validation_families: Mapping[str, str],
    training_skus: set[str],
    training_families: set[str],
    pass_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build disjointness-checked development views from frozen k=8 evidence."""

    order = list(validation_order)
    if not order or len(order) != len(set(order)):
        raise ValueError("validation order must contain unique SKU IDs")
    if set(order) != set(validation_families) or set(order) != set(pass_counts):
        raise ValueError("validation family and difficulty membership must match exactly")
    if set(order) & training_skus:
        raise ValueError("development and training SKU sets overlap")
    validation_family_set = set(validation_families.values())
    overlap = validation_family_set & training_families
    if overlap:
        raise ValueError("development and training family sets overlap")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 8
           for value in pass_counts.values()):
        raise ValueError("difficulty pass counts must be integers from zero through eight")

    bands = {
        "difficult_0_to_2_of_8": (0, 2),
        "middle_3_to_5_of_8": (3, 5),
        "easy_retention_6_to_8_of_8": (6, 8),
    }
    views: dict[str, Any] = {
        "representative_all": {
            "interpretation_allowed": len(order) >= MINIMUM_INTERPRETABLE_ROWS,
            "ordered_sku_sha256": _ordered_hash(order),
            "rows": len(order),
            "sku_ids_in_source_order": order,
        }
    }
    covered: list[str] = []
    for name, (lower, upper) in bands.items():
        members = [sku for sku in order if lower <= pass_counts[sku] <= upper]
        covered.extend(members)
        views[name] = {
            "definition": f"starting-policy whole-record pass count in [{lower},{upper}] of 8",
            "interpretation_allowed": len(members) >= MINIMUM_INTERPRETABLE_ROWS,
            "minimum_rows_for_interpretation": MINIMUM_INTERPRETABLE_ROWS,
            "ordered_sku_sha256": _ordered_hash(members),
            "rows": len(members),
            "sku_ids_in_source_order": members,
        }
    if set(covered) != set(order) or len(covered) != len(order):
        raise RuntimeError("difficulty views do not partition development rows exactly once")

    histogram = Counter(pass_counts.values())
    return {
        "source_rows": len(order),
        "source_families": len(validation_family_set),
        "difficulty_pass_count_histogram": {
            str(value): histogram[value] for value in range(9)
        },
        "views": views,
        "invariants": {
            "all_views_are_pre_outcome": True,
            "difficulty_views_partition_source_exactly_once": True,
            "family_overlap_with_run2_training": 0,
            "sku_overlap_with_run2_training": 0,
            "all_named_views_meet_interpretation_floor": all(
                view["interpretation_allowed"] for view in views.values()
            ),
        },
    }


def build_production_manifest(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    paths = {
        "sft_source": root / "data/train_weak.jsonl",
        "sft_split": root / "data/splits/sft-v1.json",
        "run2_pool": root / "data/train_weak_grpo_cap4_sft_train_v1.jsonl",
        "run2_pool_manifest": root / "runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json",
        "difficulty_manifest": root / "runs/sft-difficulty-k8/manifest.json",
        "difficulty_rollouts": root / "runs/sft-difficulty-k8/rollouts.jsonl.gz",
        "boundary_audit": root / "runs/grpo-run2-data-boundary-audit.json",
        "probe_100": root / "data/probe_100.jsonl",
        "legacy_frozen_300": root / "data/eval_300/eval.jsonl",
    }
    identities = {name: _identity(path, root) for name, path in paths.items()}
    split = json.loads(paths["sft_split"].read_text())
    pool_manifest = json.loads(paths["run2_pool_manifest"].read_text())
    difficulty_manifest = json.loads(paths["difficulty_manifest"].read_text())
    boundary = json.loads(paths["boundary_audit"].read_text())

    if identities["sft_source"]["sha256"] != split["source_sha256"]:
        raise RuntimeError("SFT source hash disagrees with split manifest")
    if identities["run2_pool"]["sha256"] != pool_manifest["output"]["active_dataset_sha256"]:
        raise RuntimeError("Run 2 pool hash disagrees with pool manifest")
    if identities["difficulty_rollouts"]["sha256"] != difficulty_manifest["artifacts"]["rollouts_sha256"]:
        raise RuntimeError("difficulty rollout hash disagrees with difficulty manifest")
    if split.get("version") != "sft-v1" or len(split["validation"]) != 360:
        raise ValueError("authoritative SFT validation contract drifted")
    if pool_manifest["output"]["active_dataset_rows"] != 1_438:
        raise ValueError("corrected Run 2 training denominator drifted")

    sft_rows = read_jsonl(paths["sft_source"])
    rows_by_sku = {row.sku_id: row for row in sft_rows}
    if len(rows_by_sku) != len(sft_rows):
        raise ValueError("SFT source contains duplicate SKU IDs")
    validation_order = list(split["validation"])
    validation_families = {
        sku: group_key(rows_by_sku[sku]) for sku in validation_order
    }
    pool_rows = read_jsonl(paths["run2_pool"])
    training_skus = {row.sku_id for row in pool_rows}
    training_families = {group_key(row) for row in pool_rows}

    rollouts: dict[str, list[bool]] = defaultdict(list)
    with gzip.open(paths["difficulty_rollouts"], "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            sku = row.get("sku_id")
            if sku not in validation_families:
                continue
            passed = row.get("passed")
            if not isinstance(passed, bool):
                raise TypeError(f"difficulty rollout {line_number} has invalid passed value")
            rollouts[sku].append(passed)
    if set(rollouts) != set(validation_order) or any(len(values) != 8 for values in rollouts.values()):
        raise ValueError("validation difficulty evidence is not exactly k=8")
    pass_counts = {sku: sum(values) for sku, values in rollouts.items()}
    development = build_development_views(
        validation_order=validation_order,
        validation_families=validation_families,
        training_skus=training_skus,
        training_families=training_families,
        pass_counts=pass_counts,
    )

    findings = boundary["headline_findings"]
    if findings["run1_validation_sku_count"] != 21:
        raise ValueError("historical Run 1 validation-use count drifted")
    if findings["probe_grpo_pool_family_count"] != 37:
        raise ValueError("probe family-overlap count drifted")

    return {
        "version": VERSION,
        "status": "development_roles_locked_confirmation_required",
        "role": "pre_inference_data_role_decision",
        "inputs": identities,
        "dataset_dispositions": {
            "corrected_run2_pool": "training",
            "sft_validation_360": "development_with_disclosed_prior_use",
            "probe_100": "not_selected_as_whole_dataset_due_training_family_overlap_and_small_size",
            "legacy_frozen_300": "legacy_reporting_only",
            "new_untouched_confirmation": "required_but_not_yet_supplied",
        },
        "training": {
            "dataset": identities["run2_pool"],
            "families": len(training_families),
            "rows": len(training_skus),
        },
        "development": {
            **development,
            "dataset": "authoritative sft-v1 validation",
            "limitations": [
                "used previously for SFT validation and checkpoint comparison",
                "21 rows were trained historically by GRPO Run 1, but neither Run 2 arm starts from the Run 1 adapter",
                "development evidence may select checkpoints but cannot support final confirmation claims",
            ],
        },
        "excluded_or_legacy": {
            "probe_100": {
                "exact_sku_overlap_with_old_grpo_pool": 0,
                "family_overlap_with_old_grpo_pool": 37,
                "rows": 100,
                "reason": "whole dataset fails the family-disjoint requirement and is too small for sole rare-class confirmation",
            },
            "legacy_frozen_300": {
                "allowed_use": "legacy reporting only",
                "reason": "already exposed during Run 1 diagnosis and therefore burned for model selection or confirmation",
                "rows": 300,
            },
        },
        "final_confirmation": {
            "assigned": False,
            "labels_opened_for_run2_selection": False,
            "model_outputs_generated": False,
            "requirements": [
                "newly sampled and frozen before final recipe selection",
                "zero SKU and normalized-family overlap with all training arms",
                "provenance and human-correction status recorded",
                "sufficient class support for macro-F1 interpretation",
            ],
        },
        "phase_e_gate": {
            "development_roles_locked": True,
            "development_training_family_overlap_zero": True,
            "development_training_sku_overlap_zero": True,
            "final_confirmation_assigned": False,
            "passed": False,
            "blocking_reason": "an untouched final-confirmation dataset has not been supplied and frozen",
        },
        "execution_boundary": {
            "gpu_training_authorized": False,
            "model_inference_performed": False,
            "phase_f_monitoring_smoke_authorized": False,
            "run2_training_contract_locked": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output = (root / args.output).resolve()
    expected = (root / DEFAULT_OUTPUT).resolve()
    if output != expected:
        raise ValueError(f"output path must be exactly {expected}")
    if output.exists():
        raise FileExistsError(output)
    artifact = build_production_manifest(root)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": args.output, "status": artifact["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
