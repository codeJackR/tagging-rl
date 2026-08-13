"""Fail-closed permission gate for Run 2 confirmation acquisition.

An HTTP 200 response proves only that an endpoint is reachable.  This module
separately proves that enough candidate stores have a documented permission
basis before any product endpoint is probed or fetched.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


VERSION = "grpo-run2-confirmation-source-gate-v1"
ALLOWED_DECISIONS = frozenset({"approved", "prohibited", "unresolved"})
DEFAULT_MINIMUM_APPROVED_STORES = 8
DEFAULT_AUDIT = "runs/grpo-run2-confirmation-terms-audit.json"
REQUIRED_PERMISSION_SCOPES = frozenset(
    {
        "automated_products_json_access",
        "research_retention",
        "human_labeling",
        "model_evaluation",
    }
)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_permission_basis(value: Any, *, domain: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"approved candidate {domain} permission_basis must be an object")
    if value.get("evidence_type") != "written_merchant_authorization":
        raise ValueError(
            f"approved candidate {domain} permission_basis has invalid evidence_type"
        )
    _required_text(value.get("evidence_reference"), f"approved candidate {domain} evidence_reference")
    evidence_hash = _required_text(
        value.get("evidence_sha256"), f"approved candidate {domain} evidence_sha256"
    ).lower()
    if len(evidence_hash) != 64 or any(character not in "0123456789abcdef" for character in evidence_hash):
        raise ValueError(f"approved candidate {domain} evidence_sha256 must be a SHA-256")
    if value.get("authorized_domain") != domain:
        raise ValueError(f"approved candidate {domain} permission domain does not match")
    _required_text(value.get("granted_at_utc"), f"approved candidate {domain} granted_at_utc")
    _required_text(value.get("grantor_role"), f"approved candidate {domain} grantor_role")
    scopes = value.get("scopes")
    if not isinstance(scopes, list) or any(not isinstance(scope, str) for scope in scopes):
        raise ValueError(f"approved candidate {domain} permission scopes must be a string list")
    missing = REQUIRED_PERMISSION_SCOPES.difference(scopes)
    if missing:
        raise ValueError(
            f"approved candidate {domain} permission scopes missing {sorted(missing)}"
        )
    if len(scopes) != len(set(scopes)):
        raise ValueError(f"approved candidate {domain} permission scopes contain duplicates")
    publication = value.get("raw_metadata_publication")
    if publication not in {"allowed", "not_allowed"}:
        raise ValueError(
            f"approved candidate {domain} raw_metadata_publication must be allowed or not_allowed"
        )
    return dict(value)


def evaluate_source_audit(
    audit: Mapping[str, Any],
    *,
    minimum_approved_stores: int = DEFAULT_MINIMUM_APPROVED_STORES,
) -> dict[str, Any]:
    """Validate an audit and decide whether product-endpoint access may begin."""

    if minimum_approved_stores <= 0:
        raise ValueError("minimum_approved_stores must be positive")
    if audit.get("version") != "grpo-run2-confirmation-terms-audit-v1":
        raise ValueError("unexpected confirmation terms-audit version")

    candidates = audit.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("audit.candidates must be a non-empty list")

    domains: set[str] = set()
    decision_counts: Counter[str] = Counter()
    approved_domains: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate {index} must be an object")
        domain = _required_text(candidate.get("endpoint_domain"), f"candidate {index} endpoint_domain")
        if domain in domains:
            raise ValueError(f"duplicate endpoint domain: {domain}")
        domains.add(domain)

        decision = _required_text(candidate.get("decision"), f"candidate {domain} decision")
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"candidate {domain} has unknown decision {decision!r}")
        _required_text(candidate.get("terms_url"), f"candidate {domain} terms_url")
        _required_text(candidate.get("evidence_summary"), f"candidate {domain} evidence_summary")
        _required_text(candidate.get("reviewed_at_utc"), f"candidate {domain} reviewed_at_utc")
        decision_counts[decision] += 1

        if decision == "approved":
            _validate_permission_basis(candidate.get("permission_basis"), domain=domain)
            approved_domains.append(domain)
        elif candidate.get("permission_basis") not in (None, ""):
            raise ValueError(
                f"non-approved candidate {domain} must not claim a permission basis"
            )

    declared = audit.get("counts")
    observed_counts = {
        "candidate_domains": len(candidates),
        "approved": decision_counts["approved"],
        "prohibited": decision_counts["prohibited"],
        "unresolved": decision_counts["unresolved"],
    }
    if declared != observed_counts:
        raise ValueError(
            f"declared source-audit counts drifted: {declared!r} != {observed_counts!r}"
        )

    passed = len(approved_domains) >= minimum_approved_stores
    return {
        "version": VERSION,
        "passed": passed,
        "minimum_approved_stores": minimum_approved_stores,
        "approved_store_count": len(approved_domains),
        "approved_domains": sorted(approved_domains),
        "decision_counts": dict(sorted(observed_counts.items())),
        "product_endpoint_requests_authorized": passed,
        "failure_reason": None
        if passed
        else (
            "published permission evidence exists for only "
            f"{len(approved_domains)} stores; contract requires at least "
            f"{minimum_approved_stores}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument(
        "--minimum-approved-stores",
        type=int,
        default=DEFAULT_MINIMUM_APPROVED_STORES,
    )
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    result = evaluate_source_audit(
        audit,
        minimum_approved_stores=args.minimum_approved_stores,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
