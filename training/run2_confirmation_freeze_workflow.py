"""Production loader and command for the final sealed Run 2 confirmation freeze."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from labeling.records import read_jsonl
from training.audit_data_boundaries import sha256_file
from training.run2_confirmation_freeze import freeze_confirmation_bundle
from training.run2_confirmation_source_gate import evaluate_source_audit
from training.split_sft import group_key
from verifier import load_pack


VERSION = "grpo-run2-confirmation-freeze-workflow-v1"
DEFAULT_TERMS_AUDIT = "runs/grpo-run2-confirmation-terms-audit.json"
DEFAULT_PRELABEL_DIR = "data/confirmation_run2_v1_prelabel"
DEFAULT_FRONTIER_DIR = "data/confirmation_run2_v1_frontier"
DEFAULT_REVIEWED_DIR = "data/confirmation_run2_v1_reviewed"
DEFAULT_PRIOR = "data/raw/labeled.jsonl"
DEFAULT_PACK = "packs/vastraa_taste_v1"
DEFAULT_PARENT_ROLE = "runs/grpo-run2-data-role-manifest.json"
DEFAULT_OUTPUT = "data/confirmation_run2_v1"
DEFAULT_ROLE_OUTPUT = "runs/grpo-run2-data-role-manifest-confirmation-assigned.json"
CODE_FILES = (
    "training/run2_confirmation_source_gate.py",
    "training/run2_confirmation_acquisition.py",
    "training/run2_confirmation_labeling.py",
    "training/run2_confirmation_labeling_workflow.py",
    "training/run2_confirmation_review.py",
    "training/run2_confirmation_review_workflow.py",
    "training/run2_confirmation_freeze.py",
    "training/run2_confirmation_freeze_workflow.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _identity(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        display = str(path.relative_to(root.resolve()))
    except ValueError:
        display = str(path)
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def committed_code_context(
    root: Path, *, code_files: Sequence[str] = CODE_FILES
) -> dict[str, Any]:
    """Require every freeze implementation file to be tracked and unchanged."""

    root = root.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("invalid Git commit identity")
    for relative in code_files:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=root,
            check=True,
            capture_output=True,
        )
    changed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *code_files],
        cwd=root,
    )
    if changed.returncode != 0:
        raise RuntimeError("confirmation freeze code differs from the recorded Git commit")
    return {
        "version": VERSION,
        "git_commit": commit,
        "files": {
            relative: _identity(root / relative, root) for relative in code_files
        },
    }


def run_freeze_workflow(
    *,
    root: Path,
    terms_audit_path: Path,
    prelabel_dir: Path,
    frontier_dir: Path,
    reviewed_dir: Path,
    prior_path: Path,
    pack_path: Path,
    parent_role_path: Path,
    output_dir: Path,
    role_output_path: Path,
    code_identity: dict[str, Any],
    frozen_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load every real artifact, verify identities, and delegate the sealed freeze."""

    root = root.resolve()
    paths = {
        "source_terms_audit": terms_audit_path.resolve(),
        "acquisition_manifest": (prelabel_dir / "acquisition-manifest.json").resolve(),
        "selection_manifest": (prelabel_dir / "selection-manifest.json").resolve(),
        "frontier_labeling_manifest": (frontier_dir / "manifest.json").resolve(),
        "review_manifest": (reviewed_dir / "manifest.json").resolve(),
        "reviewed_dataset": (reviewed_dir / "reviewed.jsonl").resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing freeze lineage {name}: {path}")
    source_audit = _read_json(paths["source_terms_audit"])
    source_gate = evaluate_source_audit(source_audit)
    if not source_gate["passed"]:
        raise PermissionError(source_gate["failure_reason"])
    selection = _read_json(paths["selection_manifest"])
    review = _read_json(paths["review_manifest"])
    support_path = (reviewed_dir / "support.json").resolve()
    support = _read_json(support_path)
    reviewed_rows = read_jsonl(paths["reviewed_dataset"])
    prior_rows = read_jsonl(prior_path)
    prior_skus = frozenset(row.sku_id for row in prior_rows)
    prior_families = frozenset(group_key(row) for row in prior_rows)
    parent = _read_json(parent_role_path)
    pack = load_pack(pack_path)
    lineage = {name: _identity(path, root) for name, path in paths.items()}
    return freeze_confirmation_bundle(
        output_dir=output_dir,
        role_output=role_output_path,
        reviewed_rows=reviewed_rows,
        selection_manifest=selection,
        review_manifest=review,
        support_report=support,
        source_gate_result=source_gate,
        prior_sku_ids=prior_skus,
        prior_family_keys=prior_families,
        pack=pack,
        lineage=lineage,
        parent_role_manifest=parent,
        parent_role_identity=_identity(parent_role_path, root),
        code_identity=code_identity,
        frozen_at_utc=frozen_at_utc,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--terms-audit", default=DEFAULT_TERMS_AUDIT)
    parser.add_argument("--prelabel-dir", default=DEFAULT_PRELABEL_DIR)
    parser.add_argument("--frontier-dir", default=DEFAULT_FRONTIER_DIR)
    parser.add_argument("--reviewed-dir", default=DEFAULT_REVIEWED_DIR)
    parser.add_argument("--prior", default=DEFAULT_PRIOR)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    parser.add_argument("--parent-role", default=DEFAULT_PARENT_ROLE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--role-output", default=DEFAULT_ROLE_OUTPUT)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    code = committed_code_context(root)
    manifest, role = run_freeze_workflow(
        root=root,
        terms_audit_path=root / args.terms_audit,
        prelabel_dir=root / args.prelabel_dir,
        frontier_dir=root / args.frontier_dir,
        reviewed_dir=root / args.reviewed_dir,
        prior_path=root / args.prior,
        pack_path=root / args.pack,
        parent_role_path=root / args.parent_role,
        output_dir=root / args.output,
        role_output_path=root / args.role_output,
        code_identity=code,
        frozen_at_utc=_utc_now(),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "role_status": role["status"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
