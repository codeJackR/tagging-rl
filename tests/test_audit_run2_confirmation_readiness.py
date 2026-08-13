import json
from pathlib import Path

import pytest

from training.audit_run2_confirmation_readiness import (
    EXPECTED,
    IMPLEMENTATIONS,
    build_readiness_report,
)


def _audit(*, approved: int = 0) -> dict:
    candidates = []
    for index in range(8):
        decision = "approved" if index < approved else "prohibited"
        candidate = {
            "endpoint_domain": f"store-{index}.example",
            "terms_url": f"https://store-{index}.example/terms",
            "reviewed_at_utc": "2026-08-13T00:00:00Z",
            "decision": decision,
            "permission_basis": {
                "evidence_type": "written_merchant_authorization",
                "evidence_reference": f"secure-record-{index}",
                "evidence_sha256": f"{index:064x}",
                "authorized_domain": f"store-{index}.example",
                "granted_at_utc": "2026-08-13T00:00:00Z",
                "grantor_role": "merchant owner",
                "scopes": [
                    "automated_products_json_access",
                    "research_retention",
                    "human_labeling",
                    "model_evaluation",
                ],
                "raw_metadata_publication": "not_allowed",
            }
            if decision == "approved"
            else None,
            "evidence_summary": "test evidence",
        }
        candidates.append(candidate)
    return {
        "version": "grpo-run2-confirmation-terms-audit-v1",
        "counts": {
            "candidate_domains": 8,
            "approved": approved,
            "prohibited": 8 - approved,
            "unresolved": 0,
        },
        "candidates": candidates,
    }


def _root(tmp_path: Path) -> tuple[Path, Path]:
    for relative in IMPLEMENTATIONS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    audit_path = tmp_path / "runs/audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    return tmp_path, audit_path


def test_blocked_permission_is_not_misreported_as_dataset_progress(tmp_path):
    root, audit_path = _root(tmp_path)
    audit = _audit(approved=0)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    report = build_readiness_report(
        root=root,
        source_audit=audit,
        source_audit_path=audit_path,
    )

    assert report["status"] == "blocked_before_acquisition_by_source_permission"
    assert report["counts"]["completed_real_stages"] == 0
    assert report["counts"]["approved_sources"] == 0
    assert all(not stage["exists"] for stage in report["real_production_stages"].values())
    assert all(value is False for value in report["execution_boundary"].values())


def test_passed_gate_advances_only_to_first_missing_real_stage(tmp_path):
    root, audit_path = _root(tmp_path)
    audit = _audit(approved=8)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    report = build_readiness_report(
        root=root,
        source_audit=audit,
        source_audit_path=audit_path,
    )

    assert report["status"] == "incomplete_at_prelabel"
    assert report["completed_real_stage_names"] == []


def test_valid_stage_statuses_are_counted_in_order(tmp_path):
    root, audit_path = _root(tmp_path)
    audit = _audit(approved=8)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    for name in ("prelabel", "frontier"):
        path = root / EXPECTED[name]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": EXPECTED[name]["status"]}), encoding="utf-8")

    report = build_readiness_report(
        root=root,
        source_audit=audit,
        source_audit_path=audit_path,
    )

    assert report["status"] == "incomplete_at_review"
    assert report["completed_real_stage_names"] == ["prelabel", "frontier"]


def test_wrong_manifest_status_does_not_count_as_completion(tmp_path):
    root, audit_path = _root(tmp_path)
    audit = _audit(approved=8)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    path = root / EXPECTED["prelabel"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "wrong"}), encoding="utf-8")

    report = build_readiness_report(
        root=root,
        source_audit=audit,
        source_audit_path=audit_path,
    )

    assert report["status"] == "incomplete_at_prelabel"
    assert report["real_production_stages"]["prelabel"]["exists"] is True
    assert report["real_production_stages"]["prelabel"]["valid_status"] is False


def test_missing_implementation_fails_closed(tmp_path):
    root, audit_path = _root(tmp_path)
    audit = _audit(approved=0)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    (root / IMPLEMENTATIONS[0]).unlink()

    with pytest.raises(FileNotFoundError, match="missing confirmation implementation"):
        build_readiness_report(
            root=root,
            source_audit=audit,
            source_audit_path=audit_path,
        )
