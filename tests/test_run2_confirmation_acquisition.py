from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from training.run2_confirmation_acquisition import (
    acquire_confirmation_candidates,
    main,
    publish_prelabel_bundle,
)
from verifier import load_pack


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pack():
    return load_pack(ROOT / "packs" / "vastraa_taste_v1")


def _audit(decisions: list[str]) -> dict:
    candidates = []
    for index, decision in enumerate(decisions):
        candidate = {
            "endpoint_domain": f"store-{index:02d}.example",
            "terms_url": f"https://store-{index:02d}.example/terms",
            "reviewed_at_utc": "2026-08-13T00:00:00Z",
            "decision": decision,
            "evidence_summary": f"Synthetic {decision} evidence.",
            "permission_basis": None,
        }
        if decision == "approved":
            candidate["permission_basis"] = {
                "evidence_type": "written_merchant_authorization",
                "evidence_reference": f"secure-record-{index}",
                "evidence_sha256": f"{index:064x}",
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
        candidates.append(candidate)
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


def _clock() -> callable:
    ticks = iter(f"2026-08-13T00:00:{second:02d}Z" for second in range(60))
    return lambda: next(ticks)


def _product(domain: str, number: int) -> dict:
    return {
        "id": number,
        "title": f"Dress {domain} {number}",
        "body_html": "<p>Woven cotton midi dress.</p>",
        "vendor": f"Brand {domain} {number}",
        "product_type": "Dress",
        "tags": ["dress", "cotton"],
        "images": [],
    }


def _row(store: int, number: int) -> dict:
    domain = f"store-{store:02d}.example"
    return {
        "sku_id": f"shopify:{domain}:{number}",
        "source": f"shopify:{domain}",
        "input": {
            "title": f"Dress {domain} {number}",
            "description": "Woven cotton midi dress.",
            "raw_tags": ["dress", "cotton"],
            "brand": f"Brand {domain} {number}",
            "category": "Dress",
            "image_url": None,
        },
    }


def _completed_report(candidate_count: int) -> dict:
    return {
        "version": "grpo-run2-confirmation-acquisition-v1",
        "status": "candidate_snapshot_acquired",
        "hard_stop": None,
        "counts": {"candidate_rows": candidate_count},
        "execution_boundary": {
            "frontier_labeling_performed": False,
            "human_review_performed": False,
            "sft_or_grpo_model_inference_performed": False,
            "gpu_training_performed": False,
        },
    }


def test_permission_gate_fails_before_requesting_any_product_page(pack) -> None:
    calls = []

    with pytest.raises(PermissionError, match="only 0 stores"):
        acquire_confirmation_candidates(
            source_audit=_audit(["prohibited"] * 8),
            pack=pack,
            request_page=lambda domain, page: calls.append((domain, page)),
            now_utc=_clock(),
        )

    assert calls == []


def test_acquisition_is_round_robin_polite_and_auditable(pack) -> None:
    calls = []
    sleeps = []

    def request(domain: str, page: int) -> tuple[int, bytes]:
        calls.append((domain, page))
        products = [_product(domain, page)] if page == 1 else []
        return 200, json.dumps({"products": products}).encode()

    rows, report = acquire_confirmation_candidates(
        source_audit=_audit(["approved"] * 8),
        pack=pack,
        request_page=request,
        now_utc=_clock(),
        sleep=sleeps.append,
        delay_seconds=1.0,
        max_pages=2,
    )

    assert calls == [
        *((f"store-{index:02d}.example", 1) for index in range(8)),
        *((f"store-{index:02d}.example", 2) for index in range(8)),
    ]
    assert sleeps == [1.0] * 15
    assert len(rows) == 8
    assert report["status"] == "candidate_snapshot_acquired"
    assert report["counts"]["requests"] == 16
    assert all(item["response_sha256"] for item in report["requests"])
    assert report["execution_boundary"]["frontier_labeling_performed"] is False


def test_403_or_429_stops_every_store_immediately(pack) -> None:
    calls = []

    def request(domain: str, page: int) -> tuple[int, bytes]:
        calls.append((domain, page))
        return 429, b""

    rows, report = acquire_confirmation_candidates(
        source_audit=_audit(["approved"] * 8),
        pack=pack,
        request_page=request,
        now_utc=_clock(),
        sleep=lambda _seconds: None,
    )

    assert rows == []
    assert calls == [("store-00.example", 1)]
    assert report["status"] == "stopped_on_blocking_http_status"
    assert report["hard_stop"]["http_status"] == 429


def test_delay_below_contract_minimum_fails_before_network(pack) -> None:
    calls = []

    with pytest.raises(ValueError, match="at least 1.0"):
        acquire_confirmation_candidates(
            source_audit=_audit(["approved"] * 8),
            pack=pack,
            request_page=lambda domain, page: calls.append((domain, page)),
            now_utc=_clock(),
            delay_seconds=0.99,
        )

    assert calls == []


def test_prelabel_bundle_publishes_400_members_atomically(pack, tmp_path) -> None:
    candidates = [_row(store, number) for store in range(10) for number in range(90)]
    output = tmp_path / "confirmation-prelabel-v1"

    manifest = publish_prelabel_bundle(
        output_dir=output,
        candidates=candidates,
        acquisition_report=_completed_report(len(candidates)),
        prior_sku_ids=frozenset(),
        prior_family_keys=frozenset(),
        pack=pack,
        source_audit_identity={"bytes": 1, "sha256": "a" * 64},
        code_identity={"git_commit": "b" * 40},
        expected_prior_sku_count=0,
    )

    assert manifest["status"] == "confirmation_membership_frozen_before_labeling"
    assert manifest["counts"] == {
        "candidate_rows": 900,
        "selected_rows": 400,
        "selected_stores": 10,
    }
    assert {path.name for path in output.iterdir()} == {
        "acquisition-manifest.json",
        "candidates.jsonl",
        "manifest.json",
        "selected.jsonl",
        "selection-manifest.json",
    }
    selected = [json.loads(line) for line in (output / "selected.jsonl").read_text().splitlines()]
    selection = json.loads((output / "selection-manifest.json").read_text())
    assert [row["sku_id"] for row in selected] == [
        item["sku_id"] for item in selection["selected"]
    ]
    assert all("labels" not in row for row in selected)

    with pytest.raises(FileExistsError, match="already exists"):
        publish_prelabel_bundle(
            output_dir=output,
            candidates=candidates,
            acquisition_report=_completed_report(len(candidates)),
            prior_sku_ids=frozenset(),
            prior_family_keys=frozenset(),
            pack=pack,
            source_audit_identity={"bytes": 1, "sha256": "a" * 64},
            code_identity={"git_commit": "b" * 40},
            expected_prior_sku_count=0,
        )


def test_failed_selection_leaves_no_output_or_staging_directory(pack, tmp_path) -> None:
    candidates = [_row(store, number) for store in range(8) for number in range(99)]
    output = tmp_path / "too-small"

    with pytest.raises(RuntimeError, match="below predeclared minimum"):
        publish_prelabel_bundle(
            output_dir=output,
            candidates=candidates,
            acquisition_report=_completed_report(len(candidates)),
            prior_sku_ids=frozenset(),
            prior_family_keys=frozenset(),
            pack=pack,
            source_audit_identity={"bytes": 1, "sha256": "a" * 64},
            code_identity={"git_commit": "b" * 40},
            expected_prior_sku_count=0,
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_hard_stopped_acquisition_cannot_publish(pack, tmp_path) -> None:
    report = _completed_report(0)
    report["status"] = "stopped_on_blocking_http_status"
    report["hard_stop"] = {"http_status": 403}

    with pytest.raises(ValueError, match="completed candidate snapshot"):
        publish_prelabel_bundle(
            output_dir=tmp_path / "blocked",
            candidates=[],
            acquisition_report=report,
            prior_sku_ids=frozenset(),
            prior_family_keys=frozenset(),
            pack=pack,
            source_audit_identity={"bytes": 1, "sha256": "a" * 64},
            code_identity={"git_commit": "b" * 40},
            expected_prior_sku_count=0,
        )


def test_production_cli_stops_at_permission_gate_without_outputs(tmp_path, capsys) -> None:
    output = tmp_path / "prelabel"
    failure = tmp_path / "failure.json"

    exit_code = main(
        [
            "--repo-root",
            str(ROOT),
            "--source-audit",
            "runs/grpo-run2-confirmation-terms-audit.json",
            "--output",
            str(output),
            "--failure-output",
            str(failure),
        ]
    )

    assert exit_code == 2
    assert not output.exists()
    assert not failure.exists()
    assert '"product_endpoint_requests_authorized": false' in capsys.readouterr().err
