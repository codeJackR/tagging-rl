#!/usr/bin/env python3
"""Production-only D3 launcher with explicit preflight and execute modes.

Preflight verifies the four locked active-replay inputs and the unused output
path without decompressing replay evidence. Execution requires an explicit
flag, builds the complete artifact in memory, validates it, and only then
publishes to the single locked D3 path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from training.run2_analysis_orchestrator import (
    build_analysis_artifact,
    run_active_preflight,
)
from training.run2_d3_result_contract import (
    DEFAULT_OUTPUT,
    EXPECTED_ACTIVE_MANIFEST,
    EXPECTED_ACTIVE_RECORDS,
    EXPECTED_CLASS_WEIGHTS,
    EXPECTED_COMPARISON_CONTRACT,
    EXPECTED_ORDERED_ROLLOUT_KEY_SHA256,
    EXPECTED_ORDERED_SKU_SHA256,
    publish_production_d3_result,
    validate_production_d3_preflight,
    validate_production_d3_result,
)


VERSION = "grpo-run2-d3-production-launcher-v2"
DEFAULT_ACTIVE_MANIFEST = EXPECTED_ACTIVE_MANIFEST["path"]
DEFAULT_ACTIVE_RECORDS = EXPECTED_ACTIVE_RECORDS["path"]
DEFAULT_COMPARISON_CONTRACT = EXPECTED_COMPARISON_CONTRACT["path"]
DEFAULT_CLASS_WEIGHTS = EXPECTED_CLASS_WEIGHTS["path"]


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
        raise ValueError("locked D3 output escapes the repository") from exc
    return output


def _require_output_absent(output: Path, *, stage: str) -> None:
    if output.exists():
        raise FileExistsError(f"locked D3 result {stage}: {output}")


def _run_locked_preflight(root: Path) -> dict[str, Any]:
    raw = run_active_preflight(
        repo_root=root,
        manifest_path=DEFAULT_ACTIVE_MANIFEST,
        records_path=DEFAULT_ACTIVE_RECORDS,
        contract_path=DEFAULT_COMPARISON_CONTRACT,
        class_weights_path=DEFAULT_CLASS_WEIGHTS,
        test_mode=False,
    )
    return validate_production_d3_preflight(raw)


def run_production_d3_preflight_only(
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Verify the locked D3 boundary and stop before replay decompression."""
    root = Path(repo_root).resolve()
    output = _locked_result_path(root)
    _require_output_absent(output, stage="already exists")
    preflight = _run_locked_preflight(root)
    _require_output_absent(output, stage="appeared during preflight")

    lineage = preflight["lineage"]
    return {
        "version": VERSION,
        "status": "production_d3_preflight_only_passed",
        "mode": "preflight_only_no_replay_decompression",
        "locked_inputs": preflight["inputs"],
        "locked_lineage": {
            **lineage,
            "ordered_sku_hash_matches_contract": (
                lineage["ordered_sku_sha256"] == EXPECTED_ORDERED_SKU_SHA256
            ),
            "ordered_rollout_hash_matches_contract": (
                lineage["ordered_rollout_key_sha256"]
                == EXPECTED_ORDERED_ROLLOUT_KEY_SHA256
            ),
        },
        "locked_settings": preflight["settings"],
        "locked_output": {
            "path": _portable(output, root),
            "exists_before_preflight": False,
            "exists_after_preflight": False,
            "complete_contract_validation_before_publication": True,
            "exclusive_atomic_publication_required": True,
        },
        "future_execution_order": [
            "rerun and validate the same active-scope production preflight",
            "build the complete aggregate artifact in memory",
            "validate the complete artifact against the locked D3 contract",
            "publish exclusively and atomically to the one locked path",
        ],
        "selection_boundary": {
            "replay_gzip_decompressed": False,
            "replay_records_parsed": False,
            "candidate_aggregate_metrics_calculated": False,
            "acceptance_gates_applied": False,
            "candidate_rankings_calculated": False,
            "winner_selected": False,
            "artifact_published": False,
            "gpu_training_authorized": False,
        },
    }


def run_production_d3_execution(
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Explicitly build, validate and publish the locked production D3 result."""
    root = Path(repo_root).resolve()
    output = _locked_result_path(root)
    _require_output_absent(output, stage="already exists")
    _run_locked_preflight(root)
    _require_output_absent(output, stage="appeared during preflight")
    artifact = build_analysis_artifact(
        repo_root=root,
        manifest_path=DEFAULT_ACTIVE_MANIFEST,
        records_path=DEFAULT_ACTIVE_RECORDS,
        contract_path=DEFAULT_COMPARISON_CONTRACT,
        class_weights_path=DEFAULT_CLASS_WEIGHTS,
        test_mode=False,
    )
    checked = validate_production_d3_result(artifact)
    publication = publish_production_d3_result(
        repo_root=root,
        artifact=checked,
    )
    return {
        "version": VERSION,
        "status": "production_d3_executed_and_published",
        "mode": "explicit_execute_active_replay_d3_only",
        "groups": checked["lineage"]["groups"],
        "completions": checked["lineage"]["completions"],
        "publication": {
            "path": publication["path"],
            "bytes": publication["bytes"],
            "sha256": publication["sha256"],
            "published_exclusively": publication["published_exclusively"],
        },
        "selection_boundary": checked["selection_boundary"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify locked inputs/output and stop before replay decompression",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="explicitly build, validate, and atomically publish D3",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = (
        run_production_d3_preflight_only(repo_root=args.repo_root)
        if args.preflight_only
        else run_production_d3_execution(repo_root=args.repo_root)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
