#!/usr/bin/env python3
"""Production-only Gate G10 launcher with explicit preflight/execute modes.

This module verifies the six locked source artifacts and the unused result path.
Execution requires an explicit flag, streams only the full-training replay, and
publishes only the single-purpose Gate G10 result. It never analyzes the active
replay, applies Gates G1-G9, ranks candidates, or selects a winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from training.run2_analysis_orchestrator import run_preflight
from training.run2_gate_g10_collector import collect_and_calculate_gate_g10
from training.run2_gate_g10_orchestrator import stream_full_groups_once
from training.run2_gate_g10_result_contract import (
    DEFAULT_OUTPUT,
    EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
    EXPECTED_ORDERED_SKU_SHA256,
    build_production_gate_g10_result,
    publish_production_gate_g10_result,
    validate_production_gate_g10_preflight,
)


VERSION = "grpo-run2-gate-g10-production-launcher-v2"
DEFAULT_ACTIVE_MANIFEST = "runs/grpo-run2-candidate-replay-manifest.json"
DEFAULT_ACTIVE_RECORDS = "runs/grpo-run2-candidate-replay-records.jsonl.gz"
DEFAULT_COMPARISON_CONTRACT = "runs/grpo-run2-comparison-contract.json"
DEFAULT_CLASS_WEIGHTS = "runs/grpo-run2-cb-class-weights.json"
DEFAULT_FULL_MANIFEST = (
    "runs/grpo-run2-full-training-candidate-replay-manifest.json"
)
DEFAULT_FULL_RECORDS = (
    "runs/grpo-run2-full-training-candidate-replay-records.jsonl.gz"
)


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _locked_result_path(repo_root: Path) -> Path:
    output = (repo_root / DEFAULT_OUTPUT).resolve()
    try:
        output.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("locked Gate G10 output escapes the repository") from exc
    return output


def _run_locked_preflight(root: Path) -> dict[str, Any]:
    """Run and contract-validate the one production dual-scope preflight."""
    raw_preflight = run_preflight(
        repo_root=root,
        manifest_path=DEFAULT_ACTIVE_MANIFEST,
        records_path=DEFAULT_ACTIVE_RECORDS,
        contract_path=DEFAULT_COMPARISON_CONTRACT,
        class_weights_path=DEFAULT_CLASS_WEIGHTS,
        full_manifest_path=DEFAULT_FULL_MANIFEST,
        full_records_path=DEFAULT_FULL_RECORDS,
        test_mode=False,
    )
    return validate_production_gate_g10_preflight(raw_preflight)


def _require_output_absent(output: Path, *, stage: str) -> None:
    if output.exists():
        raise FileExistsError(f"locked Gate G10 result {stage}: {output}")


def _validate_streamed_lineage(
    *,
    collection: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    lineage = collection.get("lineage")
    if not isinstance(lineage, dict):
        raise TypeError("Gate G10 collection lineage must be an object")
    full = preflight["full_training_replay"]
    expected = {
        "groups": full["groups"],
        "completions": full["completions"],
        "ordered_sku_sha256": full["ordered_sku_sha256"],
        "ordered_rollout_key_sha256": full["ordered_rollout_key_sha256"],
    }
    for key, expected_value in expected.items():
        if lineage.get(key) != expected_value:
            raise ValueError(
                "streamed full-training lineage differs from production "
                f"preflight: {key}"
            )


def run_production_gate_g10_preflight_only(
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Verify production inputs and stop before replay decompression."""
    root = Path(repo_root).resolve()
    output = _locked_result_path(root)
    _require_output_absent(output, stage="already exists")
    preflight = _run_locked_preflight(root)

    # Recheck after hashing so a concurrently-created result is never ignored.
    _require_output_absent(output, stage="appeared during preflight")

    inputs = preflight["inputs"]
    full = preflight["full_training_replay"]
    return {
        "version": VERSION,
        "status": "production_gate_g10_preflight_only_passed",
        "mode": "preflight_only_no_execution_path",
        "locked_inputs": {
            "active_manifest": inputs["manifest"],
            "active_records": inputs["records"],
            "comparison_contract": inputs["comparison_contract"],
            "class_weights": inputs["class_weights"],
            "full_manifest": full["manifest"],
            "full_records": full["records"],
            "cb_extension_ledger_sha256": full["cb_extension"][
                "ordered_entry_ledger_sha256"
            ],
        },
        "locked_lineage": {
            "groups": full["groups"],
            "completions": full["completions"],
            "ordered_sku_sha256": full["ordered_sku_sha256"],
            "ordered_rollout_key_sha256": full[
                "ordered_rollout_key_sha256"
            ],
            "ordered_sku_hash_matches_contract": (
                full["ordered_sku_sha256"] == EXPECTED_ORDERED_SKU_SHA256
            ),
            "ordered_rollout_hash_matches_contract": (
                full["ordered_rollout_key_sha256"]
                == EXPECTED_ORDERED_ROLLOUT_KEY_SHA256
            ),
        },
        "locked_output": {
            "path": _portable(output, root),
            "exists_before_preflight": False,
            "exists_after_preflight": False,
            "exclusive_atomic_publication_required": True,
        },
        "intended_future_execution": [
            "rerun and validate the same production preflight",
            "open the verified full-training gzip exactly once",
            "collect U, UA and CB on the shared 3,240-product denominator",
            "validate streamed lineage against the pinned manifest",
            "build and exclusively publish the locked Gate G10 result",
        ],
        "selection_boundary": {
            "active_replay_gzip_decompressed": False,
            "full_replay_gzip_decompressed": False,
            "replay_records_parsed": False,
            "gate_g10_calculated": False,
            "gate_g10_threshold_applied": False,
            "active_candidate_aggregates_calculated": False,
            "gates_g1_through_g9_applied": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
            "artifact_published": False,
            "gpu_training_authorized": False,
        },
    }


def run_production_gate_g10_execution(
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Execute and publish only Gate G10 from the locked full-training replay."""
    root = Path(repo_root).resolve()
    output = _locked_result_path(root)
    _require_output_absent(output, stage="already exists")
    preflight = _run_locked_preflight(root)
    _require_output_absent(output, stage="appeared during preflight")

    full_records = (root / DEFAULT_FULL_RECORDS).resolve()
    groups = stream_full_groups_once(full_records)
    expected_groups = preflight["full_training_replay"]["groups"]
    collection = collect_and_calculate_gate_g10(
        groups,
        expected_groups=expected_groups,
    )
    _validate_streamed_lineage(collection=collection, preflight=preflight)
    artifact = build_production_gate_g10_result(
        preflight_report=preflight,
        collection=collection,
    )
    publication = publish_production_gate_g10_result(
        repo_root=root,
        artifact=artifact,
    )
    return {
        "version": VERSION,
        "status": "production_gate_g10_executed_and_published",
        "mode": "explicit_execute_full_training_gate_g10_only",
        "groups": artifact["lineage"]["groups"],
        "completions": artifact["lineage"]["completions"],
        "candidate_order": artifact["candidate_order"],
        "gate_summary": artifact["gate_summary"],
        "publication": {
            "path": publication["path"],
            "bytes": publication["bytes"],
            "sha256": publication["sha256"],
            "published_exclusively": publication["published_exclusively"],
        },
        "selection_boundary": artifact["selection_boundary"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify locked production inputs and stop before decompression",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="explicitly stream, calculate, and atomically publish Gate G10",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = (
        run_production_gate_g10_preflight_only(repo_root=args.repo_root)
        if args.preflight_only
        else run_production_gate_g10_execution(repo_root=args.repo_root)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
