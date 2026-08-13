from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from training.run2_confirmation_source_gate import evaluate_source_audit


ROOT = Path(__file__).resolve().parent.parent
AUDIT_PATH = ROOT / "runs/grpo-run2-confirmation-terms-audit.json"


def _candidate(number: int, decision: str) -> dict:
    candidate = {
        "endpoint_domain": f"store-{number}.example",
        "terms_url": f"https://store-{number}.example/terms",
        "reviewed_at_utc": "2026-08-13T00:00:00Z",
        "decision": decision,
        "evidence_summary": f"Synthetic {decision} evidence.",
        "permission_basis": None,
    }
    if decision == "approved":
        candidate["permission_basis"] = {
            "evidence_type": "written_merchant_authorization",
            "evidence_reference": f"secure-record-{number}",
            "evidence_sha256": f"{number:064x}",
            "authorized_domain": candidate["endpoint_domain"],
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
    return candidate


def _audit(*decisions: str) -> dict:
    candidates = [_candidate(index, decision) for index, decision in enumerate(decisions)]
    return {
        "version": "grpo-run2-confirmation-terms-audit-v1",
        "counts": {
            "candidate_domains": len(candidates),
            "approved": decisions.count("approved"),
            "prohibited": decisions.count("prohibited"),
            "unresolved": decisions.count("unresolved"),
        },
        "candidates": candidates,
    }


def test_production_audit_fails_closed_before_any_product_probe() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    result = evaluate_source_audit(audit)

    assert result["passed"] is False
    assert result["approved_store_count"] == 0
    assert result["product_endpoint_requests_authorized"] is False
    assert result["decision_counts"] == {
        "approved": 0,
        "candidate_domains": 20,
        "prohibited": 16,
        "unresolved": 4,
    }


def test_eight_documented_approvals_pass_the_locked_store_floor() -> None:
    audit = _audit(*(["approved"] * 8), "prohibited")

    result = evaluate_source_audit(audit)

    assert result["passed"] is True
    assert result["approved_store_count"] == 8
    assert result["product_endpoint_requests_authorized"] is True


def test_approved_store_requires_a_nonempty_permission_basis() -> None:
    audit = _audit(*(["approved"] * 8))
    audit["candidates"][0]["permission_basis"] = None

    with pytest.raises(ValueError, match="permission_basis"):
        evaluate_source_audit(audit)


def test_approved_store_requires_all_permission_scopes() -> None:
    audit = _audit(*(["approved"] * 8))
    audit["candidates"][0]["permission_basis"]["scopes"].remove("model_evaluation")

    with pytest.raises(ValueError, match="scopes missing"):
        evaluate_source_audit(audit)


def test_approved_store_permission_must_name_the_same_domain() -> None:
    audit = _audit(*(["approved"] * 8))
    audit["candidates"][0]["permission_basis"]["authorized_domain"] = "other.example"

    with pytest.raises(ValueError, match="domain does not match"):
        evaluate_source_audit(audit)


def test_approved_store_requires_hashed_written_evidence() -> None:
    audit = _audit(*(["approved"] * 8))
    audit["candidates"][0]["permission_basis"]["evidence_sha256"] = "not-a-hash"

    with pytest.raises(ValueError, match="must be a SHA-256"):
        evaluate_source_audit(audit)


def test_duplicate_endpoint_domain_fails_closed() -> None:
    audit = _audit("approved", "approved")
    audit["candidates"][1]["endpoint_domain"] = audit["candidates"][0]["endpoint_domain"]

    with pytest.raises(ValueError, match="duplicate endpoint domain"):
        evaluate_source_audit(audit, minimum_approved_stores=1)


def test_declared_counts_must_match_candidate_evidence() -> None:
    audit = _audit("prohibited", "unresolved")
    drifted = deepcopy(audit)
    drifted["counts"]["approved"] = 1

    with pytest.raises(ValueError, match="counts drifted"):
        evaluate_source_audit(drifted)
