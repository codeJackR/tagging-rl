"""Audit whether an untouched GRPO Run 2 confirmation source exists locally."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from labeling.records import read_jsonl
from training.split_sft import group_key


VERSION = "grpo-run2-confirmation-source-audit-v1"
DEFAULT_OUTPUT = "runs/grpo-run2-confirmation-source-audit.json"


def _identity(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _generic_sku_set(path: Path) -> tuple[set[str], int]:
    sku_ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must be a JSON object")
        sku = value.get("sku_id")
        if not isinstance(sku, str) or not sku:
            raise ValueError(f"{path}:{line_number} has invalid sku_id")
        sku_ids.append(sku)
    if len(sku_ids) != len(set(sku_ids)):
        raise ValueError(f"{path} contains duplicate SKU IDs")
    return set(sku_ids), len(sku_ids)


def audit_source_membership(
    *,
    raw_feed: set[str],
    raw_labeled: set[str],
    sft: set[str],
    probe: set[str],
    eval_candidates: set[str],
    frozen_eval: set[str],
) -> dict[str, Any]:
    """Prove whether the local labeled universe contains unallocated SKUs."""

    named = {
        "raw_feed": raw_feed,
        "raw_labeled": raw_labeled,
        "sft": sft,
        "probe": probe,
        "eval_candidates": eval_candidates,
        "frozen_eval": frozen_eval,
    }
    if any(not values for values in named.values()):
        raise ValueError("confirmation source collections must be non-empty")
    if raw_feed != raw_labeled:
        raise ValueError("raw feed and labeled universes disagree")
    if sft & probe or sft & eval_candidates or probe & eval_candidates:
        raise ValueError("SFT, probe and eval-candidate allocations overlap")
    allocated = sft | probe | eval_candidates
    if not allocated <= raw_labeled:
        raise ValueError("allocated datasets contain SKUs outside the labeled universe")
    if eval_candidates != frozen_eval:
        raise ValueError("eval candidates and frozen evaluation SKU sets disagree")
    unallocated = raw_labeled - allocated
    return {
        "raw_feed_equals_labeled_universe": True,
        "allocated_partition_is_pairwise_disjoint": True,
        "allocated_rows": len(allocated),
        "labeled_rows": len(raw_labeled),
        "unallocated_labeled_rows": len(unallocated),
        "unallocated_labeled_skus": sorted(unallocated),
        "eval_candidates_equal_burned_frozen_eval": True,
        "partition_covers_labeled_universe": allocated == raw_labeled,
    }


def build_production_audit(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    paths = {
        "raw_feed": root / "data/raw/feed.jsonl",
        "raw_labeled": root / "data/raw/labeled.jsonl",
        "sft": root / "data/train_weak.jsonl",
        "probe": root / "data/probe_100.jsonl",
        "eval_candidates": root / "data/eval_candidates.jsonl",
        "frozen_eval": root / "data/eval_300/eval.jsonl",
        "corrected_run2_pool": root / "data/train_weak_grpo_cap4_sft_train_v1.jsonl",
        "data_role_manifest": root / "runs/grpo-run2-data-role-manifest.json",
    }
    identities = {name: _identity(path, root) for name, path in paths.items()}
    sku_sets: dict[str, set[str]] = {}
    observed_counts: dict[str, int] = {}
    for name, path in paths.items():
        if name == "data_role_manifest":
            continue
        sku_sets[name], observed_counts[name] = _generic_sku_set(path)

    membership = audit_source_membership(
        raw_feed=sku_sets["raw_feed"],
        raw_labeled=sku_sets["raw_labeled"],
        sft=sku_sets["sft"],
        probe=sku_sets["probe"],
        eval_candidates=sku_sets["eval_candidates"],
        frozen_eval=sku_sets["frozen_eval"],
    )
    expected_counts = {
        "raw_feed": 4_000,
        "raw_labeled": 4_000,
        "sft": 3_600,
        "probe": 100,
        "eval_candidates": 300,
        "frozen_eval": 300,
        "corrected_run2_pool": 1_438,
    }
    if observed_counts != expected_counts:
        raise ValueError(f"local source denominators drifted: {observed_counts}")

    corrected_pool_rows = read_jsonl(paths["corrected_run2_pool"])
    probe_rows = read_jsonl(paths["probe"])
    training_families = {group_key(row) for row in corrected_pool_rows}
    probe_family_by_sku = {row.sku_id: group_key(row) for row in probe_rows}
    overlapping_families = set(probe_family_by_sku.values()) & training_families
    overlapping_probe_skus = sorted(
        sku for sku, family in probe_family_by_sku.items() if family in overlapping_families
    )
    family_disjoint_probe_skus = sorted(sku_sets["probe"] - set(overlapping_probe_skus))
    if len(overlapping_families) != 33 or len(overlapping_probe_skus) != 35:
        raise ValueError("probe/corrected-training family-overlap result drifted")

    role_manifest = json.loads(paths["data_role_manifest"].read_text())
    if role_manifest.get("status") != "development_roles_locked_confirmation_required":
        raise ValueError("data-role manifest no longer requires confirmation")
    if role_manifest["final_confirmation"]["assigned"] is not False:
        raise ValueError("data-role manifest already assigns confirmation")

    return {
        "version": VERSION,
        "status": "new_confirmation_collection_required",
        "role": "read_only_local_confirmation_source_audit",
        "inputs": identities,
        "collection_counts": observed_counts,
        "labeled_universe_partition": {
            **membership,
            "components": {
                "sft": 3_600,
                "probe": 100,
                "eval_candidates": 300,
            },
        },
        "candidate_assessments": {
            "unallocated_local_labeled_rows": {
                "eligible": False,
                "rows": membership["unallocated_labeled_rows"],
                "reason": "the 4,000-row labeled universe is exhaustively allocated",
            },
            "eval_candidates_300": {
                "eligible": False,
                "rows": 300,
                "reason": "exactly the same SKU set as the diagnosis-exposed frozen evaluation set",
            },
            "probe_100_whole": {
                "eligible": False,
                "rows": 100,
                "family_overlap_with_corrected_run2_training": len(overlapping_families),
                "probe_rows_in_overlapping_families": len(overlapping_probe_skus),
                "reason": "fails the family-disjoint requirement",
            },
            "probe_family_disjoint_remainder": {
                "eligible": False,
                "rows": len(family_disjoint_probe_skus),
                "ordered_skus": family_disjoint_probe_skus,
                "reason": "65 rows are too small for sole macro-F1 confirmation and the probe is already a named diagnostic asset",
            },
            "raw_feed_or_labeled": {
                "eligible": False,
                "rows": 4_000,
                "reason": "these are source containers for the already allocated SFT, probe and evaluation products",
            },
        },
        "decision": {
            "suitable_untouched_local_source_exists": False,
            "next_action": "collect and label a new product snapshot under a predeclared confirmation acquisition contract",
            "requirements_before_freeze": [
                "sample products without using Run-2 model outputs",
                "exclude every SKU and normalized family in training and development data",
                "record source snapshot and sampling provenance",
                "record labeler, prompt, self-consistency and human-correction provenance",
                "audit class support before generating confirmation model outputs",
                "publish a collision-protected dataset and role-manifest update with hashes",
            ],
        },
        "execution_boundary": {
            "confirmation_dataset_created": False,
            "confirmation_model_outputs_generated": False,
            "gpu_training_authorized": False,
            "model_inference_performed": False,
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
    artifact = build_production_audit(root)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": args.output, "status": artifact["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
