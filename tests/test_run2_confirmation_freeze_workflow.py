from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import training.run2_confirmation_freeze_workflow as workflow


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _approved_audit():
    candidates = []
    for index in range(8):
        domain = f"store-{index}.example"
        candidates.append(
            {
                "endpoint_domain": domain,
                "terms_url": f"https://{domain}/terms",
                "reviewed_at_utc": "2026-08-13T00:00:00Z",
                "decision": "approved",
                "evidence_summary": "written permission",
                "permission_basis": {
                    "evidence_type": "written_merchant_authorization",
                    "evidence_reference": f"secure-{index}",
                    "evidence_sha256": f"{index:064x}",
                    "authorized_domain": domain,
                    "granted_at_utc": "2026-08-13T00:00:00Z",
                    "grantor_role": "merchant owner",
                    "scopes": [
                        "automated_products_json_access",
                        "research_retention",
                        "human_labeling",
                        "model_evaluation",
                    ],
                    "raw_metadata_publication": "not_allowed",
                },
            }
        )
    return {
        "version": "grpo-run2-confirmation-terms-audit-v1",
        "counts": {"candidate_domains": 8, "approved": 8, "prohibited": 0, "unresolved": 0},
        "candidates": candidates,
    }


def _paths(tmp_path: Path, audit):
    prelabel = tmp_path / "prelabel"
    frontier = tmp_path / "frontier"
    reviewed = tmp_path / "reviewed"
    terms = tmp_path / "terms.json"
    _json(terms, audit)
    _json(prelabel / "acquisition-manifest.json", {"status": "acquired"})
    _json(prelabel / "selection-manifest.json", {"status": "selected"})
    _json(frontier / "manifest.json", {"status": "frontier"})
    _json(reviewed / "manifest.json", {"status": "reviewed"})
    (reviewed / "reviewed.jsonl").write_text("{}\n", encoding="utf-8")
    _json(reviewed / "support.json", {"target_per_attribute_status_or_value": 8})
    prior = tmp_path / "prior.jsonl"
    prior.write_text("{}\n", encoding="utf-8")
    parent = tmp_path / "parent.json"
    _json(parent, {"status": "parent"})
    pack = tmp_path / "pack"
    pack.mkdir()
    return terms, prelabel, frontier, reviewed, prior, parent, pack


def test_loader_passes_all_six_lineage_identities_to_freezer(tmp_path, monkeypatch):
    terms, prelabel, frontier, reviewed, prior, parent, pack = _paths(tmp_path, _approved_audit())
    fake_prior = [SimpleNamespace(sku_id="old", input=SimpleNamespace(brand="B", title="T"))]
    fake_reviewed = [SimpleNamespace(sku_id="new")]
    monkeypatch.setattr(
        workflow,
        "read_jsonl",
        lambda path: fake_reviewed if Path(path).name == "reviewed.jsonl" else fake_prior,
    )
    monkeypatch.setattr(workflow, "group_key", lambda row: f"family:{row.sku_id}")
    monkeypatch.setattr(workflow, "load_pack", lambda path: "PACK")
    captured = {}

    def fake_freeze(**kwargs):
        captured.update(kwargs)
        return {"status": "frozen"}, {"status": "assigned"}

    monkeypatch.setattr(workflow, "freeze_confirmation_bundle", fake_freeze)
    result = workflow.run_freeze_workflow(
        root=tmp_path,
        terms_audit_path=terms,
        prelabel_dir=prelabel,
        frontier_dir=frontier,
        reviewed_dir=reviewed,
        prior_path=prior,
        pack_path=pack,
        parent_role_path=parent,
        output_dir=tmp_path / "confirmation_run2_v1",
        role_output_path=tmp_path / "assigned.json",
        code_identity={"git_commit": "a" * 40},
        frozen_at_utc="2026-08-13T00:00:00Z",
    )
    assert result == ({"status": "frozen"}, {"status": "assigned"})
    assert set(captured["lineage"]) == {
        "source_terms_audit",
        "acquisition_manifest",
        "selection_manifest",
        "frontier_labeling_manifest",
        "review_manifest",
        "reviewed_dataset",
    }
    assert captured["source_gate_result"]["passed"] is True
    assert captured["prior_sku_ids"] == frozenset({"old"})


def test_loader_stops_at_permission_before_freezer(tmp_path, monkeypatch):
    audit = _approved_audit()
    for candidate in audit["candidates"]:
        candidate["decision"] = "prohibited"
        candidate["permission_basis"] = None
    audit["counts"] = {"candidate_domains": 8, "approved": 0, "prohibited": 8, "unresolved": 0}
    terms, prelabel, frontier, reviewed, prior, parent, pack = _paths(tmp_path, audit)
    monkeypatch.setattr(
        workflow,
        "freeze_confirmation_bundle",
        lambda **kwargs: pytest.fail("freezer must not run"),
    )
    with pytest.raises(PermissionError, match="requires at least 8"):
        workflow.run_freeze_workflow(
            root=tmp_path,
            terms_audit_path=terms,
            prelabel_dir=prelabel,
            frontier_dir=frontier,
            reviewed_dir=reviewed,
            prior_path=prior,
            pack_path=pack,
            parent_role_path=parent,
            output_dir=tmp_path / "confirmation_run2_v1",
            role_output_path=tmp_path / "assigned.json",
            code_identity={},
            frozen_at_utc="2026-08-13T00:00:00Z",
        )


def test_code_context_requires_tracked_unchanged_files(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    path = tmp_path / "freeze.py"
    path.write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "freeze.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    context = workflow.committed_code_context(tmp_path, code_files=("freeze.py",))
    assert len(context["git_commit"]) == 40
    path.write_text("two\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        workflow.committed_code_context(tmp_path, code_files=("freeze.py",))


def test_code_context_rejects_untracked_file(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.py").write_text("no\n", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        workflow.committed_code_context(tmp_path, code_files=("untracked.py",))
