"""Read-only readiness audit for the untouched Run 2 confirmation set.

The report distinguishes implemented machinery from real production evidence.
It never contacts a source, invokes a labeler, opens confirmation labels to a
model, or computes a confirmation metric.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training.audit_data_boundaries import sha256_file, write_exclusive_atomic_json
from training.run2_confirmation_source_gate import evaluate_source_audit


VERSION = "grpo-run2-confirmation-readiness-v1"
DEFAULT_OUTPUT = "runs/grpo-run2-confirmation-readiness.json"

EXPECTED = {
    "prelabel": {
        "path": "data/confirmation_run2_v1_prelabel/manifest.json",
        "status": "confirmation_membership_frozen_before_labeling",
    },
    "frontier": {
        "path": "data/confirmation_run2_v1_frontier/manifest.json",
        "status": "frontier_labels_complete_awaiting_human_review",
    },
    "review": {
        "path": "data/confirmation_run2_v1_reviewed/manifest.json",
        "status": "human_review_complete_ready_for_final_freeze",
    },
    "freeze": {
        "path": "data/confirmation_run2_v1/manifest.json",
        "status": "confirmation_frozen_sealed_before_final_recipe_lock",
    },
    "role_successor": {
        "path": "runs/grpo-run2-data-role-manifest-confirmation-assigned.json",
        "status": "development_and_confirmation_roles_locked",
    },
}

IMPLEMENTATIONS = (
    "training/run2_confirmation_source_gate.py",
    "training/run2_confirmation_acquisition.py",
    "training/run2_confirmation_labeling.py",
    "training/run2_confirmation_labeling_workflow.py",
    "training/run2_confirmation_review.py",
    "training/run2_confirmation_review_workflow.py",
    "training/run2_confirmation_freeze.py",
    "training/run2_confirmation_freeze_workflow.py",
)


def _identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _artifact_state(root: Path, name: str, expected: Mapping[str, str]) -> dict[str, Any]:
    path = root / expected["path"]
    if not path.exists():
        return {
            "name": name,
            "expected_path": expected["path"],
            "exists": False,
            "valid_status": False,
            "observed_status": None,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    observed_status = value.get("status")
    return {
        "name": name,
        "expected_path": expected["path"],
        "exists": True,
        "valid_status": observed_status == expected["status"],
        "observed_status": observed_status,
        "identity": _identity(path, root),
    }


def build_readiness_report(
    *,
    root: Path,
    source_audit: Mapping[str, Any],
    source_audit_path: Path,
) -> dict[str, Any]:
    """Inspect the production path without changing it."""

    root = root.resolve()
    source_audit_path = source_audit_path.resolve()
    source_gate = evaluate_source_audit(source_audit)
    stages = {
        name: _artifact_state(root, name, expected)
        for name, expected in EXPECTED.items()
    }
    implementation = {}
    for relative in IMPLEMENTATIONS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing confirmation implementation: {relative}")
        implementation[relative] = _identity(path, root)

    sequence = ["prelabel", "frontier", "review", "freeze", "role_successor"]
    first_incomplete = next(
        (name for name in sequence if not stages[name]["valid_status"]),
        None,
    )
    if not source_gate["passed"]:
        status = "blocked_before_acquisition_by_source_permission"
        next_action = (
            "retain written collection/research-use permission for at least eight "
            "stores, publish a new point-in-time terms audit, and rerun the gate"
        )
    elif first_incomplete is None:
        status = "sealed_confirmation_ready_for_one_post_lock_use"
        next_action = "keep sealed until the final recipe and checkpoint are locked"
    else:
        status = f"incomplete_at_{first_incomplete}"
        next_action = {
            "prelabel": "run permission-gated acquisition and publish 400-member prelabel bundle",
            "frontier": "collect exactly five usable frontier labels for every selected SKU",
            "review": "complete primary, independent secondary, and adjudication review",
            "freeze": "run final overlap/verifier/lineage freeze",
            "role_successor": "publish the non-destructive assigned-role successor",
        }[first_incomplete]

    completed_real_stages = [
        name for name in sequence if stages[name]["valid_status"]
    ]
    return {
        "version": VERSION,
        "status": status,
        "source_audit": _identity(source_audit_path, root),
        "source_gate": source_gate,
        "real_production_stages": stages,
        "counts": {
            "approved_sources": source_gate["approved_store_count"],
            "required_approved_sources": source_gate["minimum_approved_stores"],
            "completed_real_stages": len(completed_real_stages),
            "required_real_stages": len(sequence),
        },
        "completed_real_stage_names": completed_real_stages,
        "implementation_identities": implementation,
        "next_action": next_action,
        "execution_boundary": {
            "network_requests_performed_by_audit": False,
            "frontier_labeling_performed_by_audit": False,
            "human_review_performed_by_audit": False,
            "model_predictions_generated_by_audit": False,
            "confirmation_metrics_calculated_by_audit": False,
            "gpu_work_performed_by_audit": False,
        },
        "interpretation": (
            "implemented and synthetically tested does not mean the real confirmation "
            "dataset exists; only valid production artifacts count as completed stages"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--source-audit",
        default="runs/grpo-run2-confirmation-terms-audit.json",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    audit_path = (root / args.source_audit).resolve()
    output = (root / args.output).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    report = build_readiness_report(
        root=root,
        source_audit=audit,
        source_audit_path=audit_path,
    )
    write_exclusive_atomic_json(output, report)
    print(json.dumps({"output": str(output), "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
