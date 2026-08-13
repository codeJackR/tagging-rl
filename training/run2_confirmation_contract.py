"""Executable pre-collection contract for Run 2 final confirmation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "grpo-run2-confirmation-acquisition-contract-v1"
DEFAULT_OUTPUT = "runs/grpo-run2-confirmation-acquisition-contract.json"
TARGET_ROWS = 400
MINIMUM_CLEAN_CANDIDATES = 800
SELECTION_SEED = 20260813
REVIEW_AUDIT_ROWS = 40

LOCKED_INPUTS = {
    "candidate_domains": {
        "path": "tools/shopify_candidates.txt",
        "bytes": 957,
        "sha256": "0c97224f9693f8346e9b1900aa4fbe2639f47dce1ae3b9c087d74b450a5eed0b",
    },
    "working_domains_point_in_time": {
        "path": "tools/shopify_stores.txt",
        "bytes": 455,
        "sha256": "8818eff765aa0ad8d34118f4842d87ce3a572d871bb3d2d2fe00f801c30d695f",
    },
    "pack_vocab": {
        "path": "packs/vastraa_taste_v1/vocab.yaml",
        "bytes": 48449,
        "sha256": "f98eba2177343867ce7f010b0ef4ec8d745b1154219a7137d29ad1807925c17d",
    },
    "pack_rules": {
        "path": "packs/vastraa_taste_v1/rules.yaml",
        "bytes": 6929,
        "sha256": "2d5e186fb5157bd1f371aa2009a02052457d49b94e7c3ef0343ec43632157c56",
    },
    "data_role_manifest": {
        "path": "runs/grpo-run2-data-role-manifest.json",
        "bytes": 44495,
        "sha256": "d67c617dddded099b2e6850012592cf1b0a4e6dcd9c5b17f681b35cfaef5ac26",
    },
    "local_source_audit": {
        "path": "runs/grpo-run2-confirmation-source-audit.json",
        "bytes": 7529,
        "sha256": "f39a6a5b168569dcac9e4799ef1b1981e2180c571e8b788412a914018e63372c",
    },
}


def _identity(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = root / expected["path"]
    raw = path.read_bytes()
    observed = {
        "path": expected["path"],
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if observed != expected:
        raise RuntimeError(f"locked confirmation-contract input drifted: {observed}")
    return observed


def build_contract(*, inputs: dict[str, Any]) -> dict[str, Any]:
    if set(inputs) != set(LOCKED_INPUTS):
        raise ValueError("confirmation contract input set drifted")
    if TARGET_ROWS <= 0 or MINIMUM_CLEAN_CANDIDATES < 2 * TARGET_ROWS:
        raise RuntimeError("confirmation candidate buffer must be at least twice target")
    if REVIEW_AUDIT_ROWS != TARGET_ROWS // 10:
        raise RuntimeError("independent review audit must remain exactly 10 percent")

    return {
        "version": VERSION,
        "status": "locked_before_confirmation_collection",
        "role": "predeclared_final_confirmation_acquisition_and_labeling_protocol",
        "inputs": inputs,
        "decision_use": {
            "allowed": "one final confirmatory comparison after recipe and checkpoint selection",
            "prohibited": [
                "reward selection",
                "beta selection",
                "training-arm selection",
                "checkpoint selection",
                "decoding or stopping-rule selection",
            ],
        },
        "acquisition": {
            "source": "public Shopify products.json endpoints after terms and endpoint probe",
            "minimum_delay_seconds": 1.0,
            "hard_stop_http_statuses": [403, 429],
            "apparel_filter": "locked pack garment-category aliases",
            "minimum_family_clean_candidate_rows": MINIMUM_CLEAN_CANDIDATES,
            "network_requests_performed_by_contract": False,
            "working_domains_must_be_reprobed": True,
        },
        "exclusions": {
            "universe": "all 4,000 previously labeled products",
            "exact_sku_overlap_allowed": 0,
            "normalized_family_overlap_allowed": 0,
            "family_key": "training.split_sft.group_key",
            "apply_before_selection_and_labeling": True,
        },
        "selection": {
            "target_rows": TARGET_ROWS,
            "seed": SELECTION_SEED,
            "tie_break": "SHA-256 of '20260813\\0<sku_id>', ascending",
            "uses_frontier_labels": False,
            "uses_sft_or_grpo_outputs": False,
            "maximum_rows_per_family": 4,
            "maximum_rows_per_store": 60,
            "maximum_store_share": 0.15,
            "minimum_stores": 8,
            "strata": "store plus provisional garment category from product type/title metadata",
            "membership_locked_before_labeling": True,
        },
        "size_rationale": {
            "prior_rows": 300,
            "target_rows": TARGET_ROWS,
            "approximate_uncertainty_scale_vs_prior": (300 / TARGET_ROWS) ** 0.5,
            "approximate_interval_narrowing_vs_prior": 1 - (300 / TARGET_ROWS) ** 0.5,
            "claim": "design approximation only; not a promised interval width",
        },
        "frontier_labeling": {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "prompt_version": "prelabel-v1",
            "structured_outputs": True,
            "usable_samples_per_product": 5,
            "diversity": "five existing user-turn prompt perturbations",
            "malformed_or_failed_request": "retry same SKU; never replace membership",
            "consensus": "labeling.consensus.consensus_labels",
        },
        "human_review": {
            "attributes_per_product": 15,
            "products": TARGET_ROWS,
            "cells_required": TARGET_ROWS * 15,
            "all_cells_reviewed": True,
            "independent_second_review": {
                "rows": REVIEW_AUDIT_ROWS,
                "share": REVIEW_AUDIT_ROWS / TARGET_ROWS,
                "selector": "SHA-256 of '20260813-review\\0<sku_id>', ascending",
            },
            "review_blinded_to_sft_and_grpo_outputs": True,
            "unresolved_cells_allowed": 0,
        },
        "support_policy": {
            "minimum_support_target_per_attribute_status_or_value": 8,
            "membership_may_change_after_labels": False,
            "shortfalls": "retain, disclose, and mark high-variance; do not replace rows",
            "support_report_required": True,
        },
        "freeze": {
            "expected_dataset": "data/confirmation_run2_v1/eval.jsonl",
            "expected_manifest": "data/confirmation_run2_v1/manifest.json",
            "collision_protected": True,
            "update_data_role_manifest": True,
            "model_outputs_allowed_before_final_recipe_lock": False,
            "aggregate_metrics_allowed_before_final_recipe_lock": False,
        },
        "failure_rules": {
            "requirements_may_be_weakened_after_collection": False,
            "failed_attempts_retained": True,
            "stop_on_overlap_or_lineage_failure": True,
            "stop_if_candidate_buffer_below_minimum": True,
            "stop_if_review_incomplete": True,
        },
        "execution_boundary": {
            "confirmation_dataset_created": False,
            "frontier_labeling_performed": False,
            "gpu_training_authorized": False,
            "human_review_performed": False,
            "model_inference_performed": False,
            "network_requests_performed": False,
        },
        "next_step": "implement and synthetically test exclusion plus metadata-only deterministic selection before any network fetch",
    }


def build_production_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    inputs = {name: _identity(root, expected) for name, expected in LOCKED_INPUTS.items()}
    role = json.loads((root / LOCKED_INPUTS["data_role_manifest"]["path"]).read_text())
    audit = json.loads((root / LOCKED_INPUTS["local_source_audit"]["path"]).read_text())
    if role.get("status") != "development_roles_locked_confirmation_required":
        raise ValueError("data-role manifest does not require confirmation")
    if audit.get("status") != "new_confirmation_collection_required":
        raise ValueError("local source audit does not require new collection")
    if audit["decision"]["suitable_untouched_local_source_exists"] is not False:
        raise ValueError("local source audit unexpectedly found a usable source")
    return build_contract(inputs=inputs)


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
    artifact = build_production_contract(root)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": args.output, "status": artifact["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
